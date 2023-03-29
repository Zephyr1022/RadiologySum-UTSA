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

from t5_embeds_encoder import MultimodalEncoder
from config import *

# The main idea is to expand the T5ForConditionalGeneration class to handle the image embeddings.

device = 'cuda'
TEXT_MODEL_NAME = MODEL_NAME
IMAGE_MODEL_NAME = IMAGE_NAME

class MultiModalt5(T5PreTrainedModel):  # nn.Module
    def __init__(self,config_vision,config_text):
        super().__init__(config_text) # MultiModalt5, self
        
        self.config_vision = config_vision
        self.config_text = config_text
    
        # Encoder
        self.text_model = T5ForConditionalGeneration.from_pretrained(TEXT_MODEL_NAME, config=self.config_text) 
        self.image_model = ViTModel.from_pretrained(IMAGE_MODEL_NAME, config=self.config_vision)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
        # Encoder
        self.encoder  = MultimodalEncoder(self.config_vision, self.config_text, self.image_model, self.text_model)

        # Decoder
        self.decoder = self.text_model.decoder
        self.lm_head = self.text_model.lm_head
        
        self.dropout = nn.Dropout(0.1)
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None, 
        pixel_values=None, 
        attention_mask=None, 
        decoder_input_ids=None, 
        labels=None, 
        encoder_outputs=None, 
        **kwargs,
    ):
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
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
        
        # INPUT IDS: torch.Size([24, 1]) torch.Size([24, 513, 768]) torch.Size([4, 513]) None
        # print("INPUT IDS:", input_ids.shape, encoder_outputs.last_hidden_state.shape, encoder_outputs.attentions.shape, decoder_input_ids)
        # print("PIXEL2", pixel_values.shape)
        # print(kwargs.keys())
        
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
    
    def __init__(self, dataframe, tokenizer, image_processor, source_len, summ_len):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.findings
        self.ctext = self.data.impression
        self.image = self.data.image
        #self.text = self.data['text'] # This is the complete findings 
        #self.ctext = self.data['summary'] # This is the summary of the findings
        
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
        
        # multiple image - average
        image_path = str(self.image[index])
        image_path_list = image_path.split(",")
        
        pixel_list = []
        for i in image_path_list:
            
            image_path ='/home/tongnian/data/mimic-cxr/'+ ''.join(i)
            try:
                ct_image = Image.open(image_path).convert('RGB')
                image_features = self.image_processor(ct_image, return_tensors="pt")
                pixel_list.append(image_features.pixel_values)
            except:
                print(f"Error opening image file {image_path}")
                return None
            
        avg_pixel_values = sum(pixel_list) / len(pixel_list)
        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, padding='max_length', return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, padding='max_length', return_tensors='pt',truncation=True)
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'pixel_values': avg_pixel_values # image_features.pixel_values
        }
    
    
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
    
    total_t0 = time.time()
    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode
    
    for idx, batch in enumerate(loader, 0):
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long)
        
        image = batch['pixel_values'].to(device)
        image = torch.squeeze(image, dim=1) 
        
        logits = model(input_ids = ids, attention_mask = mask, pixel_values = image, decoder_input_ids=y_ids, labels=lm_labels)
        loss = logits[0]

        train_total_loss += loss.item()
        if idx%200 == 0:
            print({"Training Loss": loss.item()})
            
        (loss / accumulation_steps).backward()
        
        # update the weights only after accumulating k small batches (steps)
        if (idx + 1) % accumulation_steps == 0: 
            optimizer.step()
            optimizer.zero_grad()
            
    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(loader)
    
    # training time end
    training_time = time.time() - total_t0
    
    # print result summaries
    print("===============================================")
    print(" Training Results ")
    print("===============================================")
    print(f"EPOCH {epoch+1:1d} TRAIN done: - loss {avg_train_loss:.5f}")


def validate(epoch, tokenizer, model, device, loader):
    
    total_t0 = time.time()
    rouge_score = evaluate.load("rouge")
    
    print("")
    print("Running Validation...")
    model.eval()
    
    total_valid_rouge = 0
    total_valid_loss = 0
    predictions = []
    actuals = []
    val_loss = []
    
    with torch.no_grad():
        
        for step, batch in enumerate(loader, 0):
            
            ids = batch['source_ids'].to(device, dtype = torch.long) # findings
            mask = batch['source_mask'].to(device, dtype = torch.long)
            
            image = batch['pixel_values'].to(device)
            image = torch.squeeze(image, dim=1) 
            
            y = batch['target_ids'].to(device, dtype = torch.long) # imp
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            logits = model(
                input_ids = ids, 
                attention_mask = mask, 
                pixel_values = image, 
                decoder_input_ids= y_ids, 
                labels= lm_labels)
            
            loss = logits[0]
                
            val_loss.append(loss)

            ###############################
            start_id = tokenizer.encode('<s>')[0]
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                pixel_values = image,
                max_length=128,
                num_beams=NUM_BEAMS,
                decoder_start_token_id=start_id,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                do_sample=True,
            )
            ###############################
            
            # Use the tokenizer to convert the output to a string
            # decoded preds	and labels
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            rouge_score.add_batch(predictions=preds, references=target)
            
            predictions.extend(preds)
            actuals.extend(target)
            
            if step%200 == 0:
                print(f'Completed {step} step...')
                
        avg_val_loss = statistics.fmean(val_loss)
        print("validation loss:", avg_val_loss)
        
        result2 = rouge_score.compute()
        rouge1_f1 = result2['rouge1']
        rouge2_f1 = result2['rouge2']
        rougel_f1 = result2['rougeL']
        
        print("--- ROUGE ---")
        print("rouge1:", rouge1_f1)
        print("rouge2:", rouge2_f1)
        print("rougeL:", rougel_f1)
        
        total_valid_rouge = (rouge1_f1+rouge2_f1+rougel_f1)/3
        
        print("")
        print("==============================================")
        print("Validation Results")
        print("==============================================")
        print("| Epoch | Val loss | ROUGE1 | ROUGE2 | ROUGE-L | Avg Rouge |")
        print(f"| {epoch+1:5d} | {avg_val_loss} | {rouge1_f1} | {rouge2_f1} | {rougel_f1} | {total_valid_rouge} |")
        
    return predictions, actuals, rougel_f1


def main():
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed

    torch.backends.cudnn.deterministic = True
        
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    tokenizer.add_tokens(['<s>'])
    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    
    config_vision = ViTConfig()
    config_text = AutoConfig.from_pretrained(TEXT_MODEL_NAME, use_cache=False)
    # config_text.gradient_checkpointing = GRADIENT_CHECKPOINT
    
    model = MultiModalt5(config_vision,config_text)
    model = model.to(device)
    print("model", model)
    
    # Importing and Pre-Processing the domain data. Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    train_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/train.json', lines=True)[['findings', 'impression','image']]#.iloc[:12]
    val_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/valid.json', lines=True)[['findings', 'impression','image']]#.iloc[:12]
    test_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/test.json', lines=True)[['findings', 'impression','image']]#.iloc[:12]

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    
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
        
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps= 1e-08, 
            weight_decay=0.,
            amsgrad=False)
        
#       optimizer = Lion(
#           params=model.parameters(),
#           lr=LEARNING_RATE,
#           betas=(0.9, 0.999),
#           weight_decay=1e-2,
#           use_triton=True # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
#       )
        
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
                exp_folder_path = "/home/tongnian/RadiologySumm/experiments/text-embeds"
                torch.save(model.state_dict(), exp_folder_path + "/best_model.pt")
                print('Best Val Rouge L Model was saved:', best_val_rouge)
                with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([predictions, actuals])
    
if __name__ == '__main__':
    main()

# option1
# text -> encoder -> encoding text + image -> decoder

# option 2
# text -> token embeddings -> token embeddings + image embedding -> encoder -> decoder
    
# option 3
# text -> token embeddings -> token embeddings + image embedding -> encoder -> encoder embeddings + original image embedding -> decoder

#Evaluate the model performance with the sumeval's Rouge score.
#Rouge1: Evaluate the generated text in units of uni-grams.
#Rouge2: Evaluate the generated text in units of bi-grams.
#RougeL: Evaluate the match of the generated text sequence.


## Defining some key variables that will be used later on in the training  
#GRADIENT_ACCUM = 8
#TRAIN_BATCH_SIZE = 2
#VALID_BATCH_SIZE = 2
#TEST_BATCH_SIZE = 2
#
## this means training will be done for affective batch size of BATCH_SIZE * GRADIENT_ACCUM
#EPOCHS = 15
#LEARNING_RATE = 1e-4
#
## random seed
#SEED = 42
#MAX_LEN = 512
#SUMMARY_LEN = 128
#MODEL_NAME = 'google/flan-t5-base' # google/flan-t5-base, google/flan-t5-large
#IMAGE_NAME = "google/vit-base-patch16-224"
#
## generation
#NUM_BEAMS = 6
#NUM_EPOCH = 15
#
#GRADIENT_CHECKPOINT = True
#gradient_checkpointing_option = True
#mixed_precision_option = 'no'