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

import torch.nn as nn
import torch
from torch import cuda

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
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack, T5PreTrainedModel
from transformers import T5Config
#from transformers.configuration_t5 import T5Config
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import GPT2Config
from transformers import GPTNeoConfig, GPTNeoModel


from t5_encoder import MultimodalEncoder
from config import *


# The main idea is to expand the T5ForConditionalGeneration class to handle the image embeddings.

device = 'cuda'

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
        image_path ='/home/tongnian/data/mimic-cxr/'+ ''.join(image_path.split(',')[0]) # only using one image
        
        # error handling
        try:
            ct_image = Image.open(image_path).convert('RGB') #.convert('RGB')
        except:
            print(f"Error opening image file {image_path}")
            return None
        
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
    
    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode
    optimizer.zero_grad() # reset gradients for accumulation for the next large_batch

    for idx, batch in enumerate(loader, 0):
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        image = batch['pixel_values'].to(device)
        image = torch.squeeze(image, dim=1) 
        
        
        # outputs = logits, loss function
        logits = model(pixel_values = image, decoder_input_ids=y_ids, labels=lm_labels)
        # loss = model(pixel_values=image, decoder_input_ids=y_ids, labels=lm_labels).loss
        loss = logits[0]
        
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
    print(f"EPOCH {epoch+1:1d} TRAIN done: - loss {avg_train_loss:.5f}")


def validate(epoch, tokenizer, model, device, loader):
    # capture validation time
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
            
            # outputs = logits, loss function
            logits = model(
                pixel_values = image, 
                decoder_input_ids=y_ids, 
                labels=lm_labels
            )

            loss = logits[0]
            val_loss.append(loss)
            
            # torch.Size([1, 128]) torch.Size([1, 128]) torch.Size([1, 3, 224, 224])
            # print("source_ids", ids.shape, "mask", mask.shape,"image", image.shape) 
            
            ###############################
            start_id = tokenizer.encode('<s>')[0]
            generated_ids = model.generate(
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
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.add_tokens(['<s>'])
    
#   config_encoder = ViTConfig()
#   config_decoder = GPT2Config()
#   config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
#   model = VisionEncoderDecoderModel(config=config)
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        'google/vit-base-patch16-224','gpt2'
    )
    
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    print("model", model)

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

            predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
            sys.stdout.flush()
            
            # Save best model based on ROUGE score\
            if val_rouge > best_val_rouge:
                best_val_rouge = val_rouge
                
                # save best model for use later
                exp_folder_path = "/home/tongnian/RadiologySumm/experiments/vision-gpt"
                torch.save(model.state_dict(), exp_folder_path + "/best_model.pt")
                print(best_val_rouge, ' Best Val Rouge L Model was saved.')
                with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([predictions, actuals])
                    
#Evaluate the model performance with the sumeval's Rouge score.
#Rouge1: Evaluate the generated text in units of uni-grams.
#Rouge2: Evaluate the generated text in units of bi-grams.
#RougeL: Evaluate the match of the generated text sequence.
    
if __name__ == '__main__':
    main()

    
    
    