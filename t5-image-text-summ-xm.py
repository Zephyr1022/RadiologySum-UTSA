# Importing libraries
import os
import logging
import random
import sys
import time
import csv
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import Adafactor, AdafactorSchedule
import transformers
from datasets import load_dataset, load_metric
import evaluate
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import optuna 
from config import *
from torch import cuda
import torch.nn as nn
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import ViTConfig, ViTModel
from transformers import AutoImageProcessor, ViTModel
from transformers import AutoModelForCausalLM, T5EncoderModel, ViTModel
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoFeatureExtractor
from transformers import AutoConfig
import inspect
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
#from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers import T5Config
#from transformers.configuration_t5 import T5Config


# The main idea is to expand the T5ForConditionalGeneration class to handle the image embeddings.

device = 'cuda'

class MultiModalt5(T5ForConditionalGeneration):  # nn.Module
    def __init__(self, configTEXT, configViT):
        super().__init__(configTEXT) # MultiModalt5, self

        self.configTEXT = configTEXT
        self.configViT = configViT
        
        self.image_embed_dim = configViT.hidden_size # 768
        self.text_embed_dim = configTEXT.hidden_size
        self.projection_dim = configTEXT.hidden_size # the number of output features or dimensions

        # Encoder x 2 
        self.text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', config=self.configTEXT) 
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=self.configViT)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Decoder
        self.visual_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        
        # self.enc_to_dec_proj = nn.Linear(self.projection_dim * 2, self.projection_dim, bias=False)
        self.enc_to_dec_proj = nn.Linear(513, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, decoder_input_ids=None, labels=None):
        
        # Encode image inputs
        image_outputs = self.image_model(pixel_values=pixel_values) # (batch_size, hidden_size)
        image_h = image_outputs[1] # pooled_output
        image_embeds = self.visual_projection(image_h) # (batch_size, projection_dim) 
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # (batch_size,)
        
        print("image_h", image_h.shape) # ([4, 768])
        print("image_embeds", image_embeds.shape) # torch.Size([4, 768])
        #print("image_atts", image_atts.shape) # torch.Size([4])

        # Encode text inputs
        text_encoder = self.text_model.get_encoder()
        text_decoder = self.text_model.get_decoder()
        last_linear_layer = self.text_model.lm_head
        
        print("text_decoder", inspect.signature(text_decoder.forward))
        print(type(text_decoder))
        
        text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask) # (batch_size, sequence_length, hidden_size)
        text_h = text_outputs.last_hidden_state # text_outputs[0]
        text_embeds = self.text_projection(text_h) # (batch_size, sequence_length, projection_dim)
        
        print("text_h", text_h.shape) # torch.Size([1, 512, 768])
        print("text_embeds", text_embeds.shape) # torch.Size([1, 512, 768])
        
        # text_atts = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(device) # (batch_size, sequence_length)
        # print("text_atts", text_atts.shape) # torch.Size([4, 512])

        # Concatenate the hidden states from text and image encoders into a joint representation
        # image_atts = image_atts.unsqueeze(1).expand(-1, text_atts.shape[1])
        # encoder_atts = torch.cat([image_atts, text_atts], dim=1) # torch.Size([2, 1024])
        
        # Repeat the image embedding tensor along the sequence length dimension
        # (batch_size, sequence_length + number of images, projection_dim)
        
        # multiple 512 times???
        # image_embeds = image_embeds.unsqueeze(1).expand(-1, text_embeds.shape[1], -1) # [1, 512, 768]
        # inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1) # (batch_size, sequence_length, projection_dim * 2)
        inputs_embeds = torch.cat((image_embeds.unsqueeze(1), text_embeds), dim=1)
        
        print("inputs_embeds", inputs_embeds.shape) # torch.Size([1, 512, 1536])
        
        # a projection layer after the concat: inputs_embeds
        # The main thing is to make sure that that is the correct size that is expected by the decoder
        # If the decoder expects the input embedding size to be hidden_size
        # input_embeds(the variable defined after the cat() method) will be 2*hidden size, not hidden_size. So, you may get an error.
        # inputs_embeds = self.enc_to_dec_proj(inputs_embeds) 
        # print("inputs_embeds_proj", inputs_embeds.shape)

        # encoder_linear_layer = nn.Linear(text_atts.shape[1] * 2, text_atts.shape[1])
        # encoder_atts = encoder_linear_layer(encoder_atts)
        # print("inputs_embeds", encoder_atts.shape)
        attention_mask = torch.cat((torch.ones((1, 1)).to(device), attention_mask), dim=1)
        print("attention_mask", attention_mask.shape)

        # Decode the concatenated hidden states using the T5 decoder -> (batch_size, sequence_length, hidden_size)
        decoder_outputs = text_decoder(input_ids=decoder_input_ids,
                        #attention_mask=mask_ids, # you will probably need this, but I need to look into it more
                        encoder_hidden_states=inputs_embeds, # Last hidden states combined with image embeds
                        encoder_attention_mask=attention_mask )
        
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
                    encoder_last_hidden_state=inputs_embeds,#encoder_outputs.last_hidden_state,
                    encoder_hidden_states=text_outputs.hidden_states,
                    encoder_attentions=text_outputs.attentions,
                )
                
    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
        ):
            decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_input_ids": decoder_inputs["input_ids"],
                "encoder_outputs": encoder_outputs,
                "past_key_values": decoder_inputs["past_key_values"],
                "use_cache": use_cache,
            }
            return input_dict
        
    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)
    


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
        
        # multiple image??? ~ one summary
        image_path = str(self.image[index])
        image_path ='/home/tongnian/data/mimic-cxr/'+ ''.join(image_path.split(',')[0])
        
        # error handling
        try:
            ct_image = Image.open(image_path).convert('RGB') #.convert('RGB')
        except:
            print(f"Error opening image file {image_path}")
            return None

        # model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        # image_features = model.get_image_features(**inputs)
        # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py

        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, padding='max_length', return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, padding='max_length', return_tensors='pt',truncation=True)
        image_features = self.image_processor(ct_image, return_tensors="pt")  # do we need to padding? 

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()        
        
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'pixel_values': image_features.pixel_values
    }
    
    
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
    total_t0 = time.time()
    print('Training...')

    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode
    optimizer.zero_grad() # reset gradients for accumulation for the next large_batch
    # for each large batch of training data...
    for idx, batch in enumerate(loader, 0):
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long)
        image = batch['pixel_values'].to(device)
        image = torch.squeeze(image, dim=1) 

        # outputs = logits, loss function
        logits = model(input_ids = ids, attention_mask = mask, pixel_values = image, decoder_input_ids=y_ids, labels=lm_labels)
        loss = logits[0]
        
        print("logits, loss:",loss)
        
        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()
        if idx%200 == 0:
            print({"Training Loss": loss.item()})

        # backpropagation-> gradient accumulation to update the model's parameters
        (loss / accumulation_steps).backward() # gradeints computed for small_batch
        
        # update the weights only after accumulating k small batches (steps)
        if (idx + 1) % accumulation_steps == 0: 
            # print(f"==> Gradient accumulation after step {accumulation_steps} in batch {idx+1}...")
            optimizer.step()
            optimizer.zero_grad() # reset gradients for accumulation for the next large_batch
    
    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(loader)
    
    # training time end
    training_time = time.time() - total_t0
    
    # print result summaries
    print("===============================================")
    print(" Training Results ")
    print("===============================================")
    print(" Epoch | average train loss |")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} |")




    
def main():
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.add_tokens(['<s>'])
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    config_text = AutoConfig.from_pretrained("google/flan-t5-base")
    config_vision = ViTConfig()
    
    model = MultiModalt5(configTEXT = config_text, configViT = config_vision)
    model = model.to(device)
    
    print("model", model)

    #model = T5EncoderModel.from_pretrained("google/flan-t5-base")
    #model = model.to(device)
    #model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    #model = model.to(device)

    # Importing and Pre-Processing the domain data. Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    train_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/train.json', lines=True)[['findings', 'impression','image']].iloc[:10]
    val_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/valid.json', lines=True)[['findings', 'impression','image']].iloc[:10]
    test_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/test.json', lines=True)[['findings', 'impression','image']].iloc[:10]

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

    print("training_set", training_loader)
    
    
    print("=================")
    print('MODE is: ', MODE)
    
    if MODE == 'train' or MODE == 'test':
        print('Start training')
        
        optimizer = torch.optim.AdamW(params=model.parameters(),
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
            model.to(device)
            
            train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
            sys.stdout.flush()

#Evaluate the model performance with the sumeval's Rouge score.
#Rouge1: Evaluate the generated text in units of uni-grams.
#Rouge2: Evaluate the generated text in units of bi-grams.
#RougeL: Evaluate the match of the generated text sequence.
    
if __name__ == '__main__':
    main()

    