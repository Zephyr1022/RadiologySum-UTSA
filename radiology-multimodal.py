# Importing libraries
import os
import logging
import random
import sys
import time
import csv
import copy
import statistics
import numpy as np
import pandas as pd
import optuna 
import gc
import nltk
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import Adafactor, AdafactorSchedule
from datasets import load_dataset, load_metric
import evaluate
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets

import transformers
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import ViTConfig, ViTModel
from transformers import AutoImageProcessor, ViTModel
from transformers import AutoModelForCausalLM, T5EncoderModel, ViTModel
from transformers import AutoFeatureExtractor
from transformers import AutoConfig

import inspect
from PIL import Image

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack, T5PreTrainedModel
from transformers import T5Config

from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from lion_pytorch import Lion

from src.x_rem_encoder import MultimodalEncoder
from src.config_match import *
from src.helper import *
from src.training import *
from src.embeddings import *

# The main idea is to expand the T5ForConditionalGeneration class to handle the image embeddings.
# text -> token embedding + image -> enc() -> encoder embeds -> decoder()
# the image embeddings can be passed as the hidden state

device = 'cuda'
exp_folder_path = "./radiology/experiments/x-rem-clinical-base"

TEXT_MODEL_NAME = MODEL_NAME
IMAGE_MODEL_NAME = IMAGE_NAME

class MultiModalt5(T5PreTrainedModel):  # nn.Module
    def __init__(self,config_vision,config_text):
        super().__init__(config_text) # MultiModalt5, self
        
        self.config_vision = config_vision
        self.config_text = config_text
    
        # Load models and tokenizer
        self.text_model = T5ForConditionalGeneration.from_pretrained(TEXT_MODEL_NAME, config=self.config_text) 
        self.image_model = ViTModel.from_pretrained(IMAGE_MODEL_NAME, config=self.config_vision)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
        # Encoder
        self.encoder  = MultimodalEncoder(self.config_vision, self.config_text, self.image_model, self.text_model)

        # Decoder
        self.decoder = self.text_model.decoder
        self.lm_head = self.text_model.lm_head
        self.dropout = nn.Dropout(0.1)
        
    # Add this method to resize token embeddings
    def resize_token_embeddings(self, new_num_tokens):
        self.text_model.resize_token_embeddings(new_num_tokens)
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None, 
        pixel_values=None,
        num_images=None,
        attention_mask=None, 
        decoder_input_ids=None, 
        labels=None, 
        encoder_outputs=None, 
        **kwargs,
    ):
        
        # torch.Size([2, 3, 3, 224, 224]) tensor([2, 1], device='cuda:0')
        # reshape to (batch_size * max_num_images) x image_x x image_y
        # print("forward", pixel_values.shape, num_images)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                num_images=num_images,
            )
        
        if encoder_outputs.last_hidden_state.shape[0] != encoder_outputs.attentions.shape[0]:
            times_to_repeat = int(encoder_outputs.last_hidden_state.shape[0]/encoder_outputs.attentions.shape[0])
            new_attention = encoder_outputs.attentions.tile((times_to_repeat, 1))
        else:
            new_attention = encoder_outputs.attentions
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state, # Last hidden states combined with image embeds
            encoder_attention_mask=new_attention
        )
        
        last_linear_layer = self.lm_head
        logits = last_linear_layer(decoder_outputs[0])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return Seq2SeqLMOutput(
                    loss=loss,
                    logits=logits,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )
    
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        decoder_input_ids = None,
        pixel_values=None,
        num_images=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        encoder_hidden_states=None,
        **kwargs
    ):
        '''
        INPUT IDS: torch.Size([24, 1]) torch.Size([24, 513, 768]) torch.Size([4, 513]) None
        print("INPUT IDS:", input_ids.shape, encoder_outputs.last_hidden_state.shape, encoder_outputs.attentions.shape, decoder_input_ids)
        print("PIXEL2", pixel_values.shape)
        print(kwargs.keys())
        '''
        
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        return {
            "decoder_input_ids": input_ids, #"input_ids": input_ids,
            # "encoder_hidden_states": encoder_outputs.hidden_states,
            # "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            # "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "pixel_values": pixel_values,
            "num_images":num_images,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past
        
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class CustomDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, image_processor, source_len, summ_len): #json
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.findings
        self.ctext = self.data.impression
        self.image = self.data.image 
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        """
            Constructs an image processor and a tokenizer into a single processor.
        """
        text = str(self.text[index])
        text = ' '.join(text.split())
        ctext = str(self.ctext[index])
        ctext = '<s> '+' '.join(ctext.split()) # df.ctext = 'summarize: ' + df.ctext/ inputs = ["summarize: " + text]
        
        # single text
        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, padding='max_length', return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, padding='max_length', return_tensors='pt',truncation=True)
        
        # squeeze() to remove the batch dimension from the input data if the batch size is 1
        # expect input tensors with shape [sequence_length] instead of [batch_size, sequence_length]
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        # multiple image - average
        image_path = str(self.image[index])
        image_path_list = image_path.split(",")
        
        # num_examples = 2 # batch size, process one example at a time
        max_images = max(len(str(i).split(",")) for i in self.image)
        images_color = 3
        image_X_dim = 224
        image_Y_dim = 224

        # Create a new tensor with the desired shape (num_examples, 3, 3, 224, 224) and fill it with zeros
        # padded_tensor = torch.zeros((num_examples, max_images, images_color, image_X_dim, image_Y_dim))
        padded_tensor = torch.zeros((max_images, images_color, image_X_dim, image_Y_dim))

        pixel_list = []
        for i in image_path_list:
            image_path ='./radiology/data/'+ ''.join(i)
#           print(image_path)
            try:
                ct_image = Image.open(image_path).convert('RGB')
                image_features = self.image_processor(ct_image, return_tensors="pt")
                pixel_list.append(image_features.pixel_values.squeeze())
            except:
                print(f"Error opening image file {image_path}")
                return None
            
        # combine pixel values from a list of images into a single tensor 
        combined_pixel_values = torch.stack(pixel_list, dim=0)
        
        # and then pad the tensor
        padded_tensor[:combined_pixel_values.size(0), :, :, :] = combined_pixel_values

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'pixel_values': padded_tensor, # image_features.pixel_values
            'num_images': torch.tensor(len(image_path_list)).to(dtype=torch.long)
        }

def main():
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    tokenizer.add_tokens(['<s>'])
    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    
    config_vision = ViTConfig()
    config_text = AutoConfig.from_pretrained(TEXT_MODEL_NAME, use_cache=False)
    
    # Inital model
    model = MultiModalt5(config_vision,config_text)
    model.resize_token_embeddings(len(tokenizer)) # RuntimeError: CUDA error: device-side assert triggered add more token <s>
    model = model.to(device)
    print("model:\n", model)
    
    # Load Dataset
    train_dataset = pd.read_json('./radiology/data/train.json', lines=True)[['findings', 'impression','image']]
    val_dataset = pd.read_json('./radiology/data/valid.json', lines=True)[['findings', 'impression','image']]
    test_dataset = pd.read_json('./radiology/data/test.json', lines=True)[['findings', 'impression','image']]

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)
    test_set = CustomDataset(test_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)
    
    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }
    
    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    test_params = {
        'batch_size': TEST_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    # Creation of Dataloaders # batch
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **val_params)


    print("=================")
    print('MODE is: ', MODE)
    print("=================")
    
    if MODE == 'train':
        
        print('Start training')

        # every weight used in a forward propagation of a neural network will be trained via gradients calculated using back propagation
        # do this before creating the optimizer so that the optimizer does not consider the frozen parameters when updating the model weights. 
        if True: # Set this to True if you don't want to train the image encoder weights
            for param in model.image_model.parameters():
                param.requires_grad = False # freezing

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps= 1e-08, 
            weight_decay=0.,
            amsgrad=False)

        # Training loop
        print('Initiating Fine-Tuning...')
        epochs = EPOCHS
        accumulation_steps = GRADIENT_ACCUM
        training_stats = []
        valid_stats = []
        best_val_rouge = 0

        for epoch in range(epochs):

            train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
            sys.stdout.flush()

            predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
            sys.stdout.flush()
            
            # Save best model based on ROUGE score
            if val_rouge > best_val_rouge:
                best_val_rouge = val_rouge
                
                # save best model for use later
                torch.save(model.state_dict(), exp_folder_path + f"/best_model.pt")
                print('Best Val Rouge L Model was saved:', best_val_rouge)
                with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([predictions, actuals])
                    
                    
    elif MODE == 'inference':
        print('Start evaluation')
        
        SAVE_PATH = "./RadiologySumm/test-results/x-rem-image"
        MODEL_PATH = exp_folder_path + f"/best_model.pt"
        print("MODEL_PATH", MODEL_PATH)
        
        print('Loading model...')
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(device)
        model.eval()
        epoch = 1
        
        # inference
        predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
        sys.stdout.flush()
        
        print('Generating output files for evaluation...')
        # Save the predictions and actuals to a CSV file for further evaluation
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv(SAVE_PATH + f"/output_valid.csv")
        print('Done.')
        
    
    elif MODE == 'embeds':
        print('Start extract embeddings')
        
        SAVE_PATH = "./data/mimic-cxr/modalities_embeds/"
        MODEL_PATH = exp_folder_path + f"/best_model.pt"
        print("MODEL_PATH", MODEL_PATH)
        
        print('Loading model...')
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(device)
        model.eval()
        epoch = 1
        
        embeddings(epoch, tokenizer, model, device, training_loader, SAVE_PATH, 'embeddings_train')
        embeddings(epoch, tokenizer, model, device, val_loader, SAVE_PATH, 'embeddings_valid')
        embeddings(epoch, tokenizer, model, device, test_loader, SAVE_PATH, 'embeddings_test')
        
        print("Done.")
        sys.stdout.flush()


if __name__ == '__main__':
    main()


    
    