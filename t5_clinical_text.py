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

from radgraph import F1RadGraph
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import optuna
from nltk.tokenize import wordpunct_tokenize

from config_t5 import *
from torch import cuda
    
device = 'cuda'
TEXT_MODEL_NAME = MODEL_NAME

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def process_impression(impression):
    impression = impression.lower()
    return ' '.join(wordpunct_tokenize(impression))

def process_predictions_and_actuals(predictions, actuals, SAVE_PATH, output_file_name):
    predictions_new = []
    for pred in predictions:
        pred = pred.replace("<s>", "")
        pred = process_impression(pred)
        predictions_new.append(pred)
        
    actuals_new = []
    for actual in actuals:
        actual = actual.replace("<s>", "")
        actual = process_impression(actual)
        actuals_new.append(actual)
        
    final_df_test = pd.DataFrame({'Generated Text': predictions_new, 'Actual Text': actuals_new})
    final_df_test.to_csv(SAVE_PATH + f"/{output_file_name}.csv")
    
    print('Generating output files for submission...')
    with open(SAVE_PATH + f"/{output_file_name}.txt", 'w') as file:
        file.write('\n'.join(predictions_new))
        
def get_text_encoder_embeds(model, input_ids, attention_mask):
    encoder_outputs = model.encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict = True,
    )
    batch_text_embeds = encoder_outputs.last_hidden_state
    
    # Extract the last token embeddings for each sequence in the batch
    last_token_indices = attention_mask.sum(dim=1) - 1
    last_token_embeds = batch_text_embeds[torch.arange(batch_text_embeds.size(0)), last_token_indices] # EOS token
    
    return last_token_embeds


class CustomDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.findings # json
        self.ctext = self.data.impression
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        
        text = str(self.text[index])
        text = ' '.join(text.split())
        
        ctext = str(self.ctext[index])
        ctext = '<s> '+' '.join(ctext.split())
        
        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, padding='max_length',return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, padding='max_length',return_tensors='pt',truncation=True)
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
    
    total_t0 = time.time() # capture time
    
    # Perform one full pass over the training set
    print("")
    print('============== Epoch {:} / {:} =============='.format(epoch + 1, epochs))
    print('Training...')

    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode
    optimizer.zero_grad() # reset gradients for accumulation for the next large_batch
    # for each large batch of training data...
    for idx, batch in enumerate(loader, 0):
        
        # progress update every 200 batches.
        if idx % 200 == 0:
            print(' ---> Batch {:>5,}  of  {:>5,}.'.format(idx, len(loader)))
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long)
        
        # outputs = logits, loss function
        logits = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
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
    print(" Epoch | average train loss |")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} |")


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
            y = batch['target_ids'].to(device, dtype = torch.long)
            ids = batch['source_ids'].to(device, dtype = torch.long)
            mask = batch['source_mask'].to(device, dtype = torch.long)
            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            
            logits = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = logits[0]
            val_loss.append(loss)
            
            ###############################
            start_id = tokenizer.encode('<s>')[0]
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=GENERATION_LEN,
                num_beams=NUM_BEAMS,
                decoder_start_token_id=start_id,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            ###############################
            
            # Use the tokenizer to convert the output to a string
            # decoded preds    and labels
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            rouge_score.add_batch(predictions=preds, references=target)
            #bertscore.add_batch(predictions=preds, references=target)
            #bleu_score.add_batch(predictions=preds, references=target)
            #score, _, predictions_lists, actuals_lists = f1radgraph(hyps=preds, refs=target)

            predictions.extend(preds)
            actuals.extend(target)

            if step%200 == 0:
                print(f'Completed {step} step...')
        
        avg_val_loss = statistics.fmean(val_loss)
        print("---validation loss:", avg_val_loss)
        
        # Compute metrics
        #result1 = bleu_score.compute()
        result2 = rouge_score.compute()
        #result3 = bertscore.compute(lang="en")

        #bleu = result1['bleu']
        #print("--- BLEU ---:", bleu)
        
        #rouge1_f1 = result2['rouge1']
        #rouge2_f1 = result2['rouge2']
        rougel_f1 = result2['rougeL']
        
        print("--- ROUGE ---")
        #print("rouge1:", rouge1_f1)
        #print("rouge2:", rouge2_f1)
        print("rougeL:", rougel_f1)
        #ave_valid_rouge = (rouge1_f1+rouge2_f1+rougel_f1)/3
        
        predictions = [str(line) for line in predictions]
        actuals = [str(line) for line in actuals]
        
        f1radgraph = F1RadGraph(reward_level="partial")
        score, _, predictions_lists, actuals_lists = f1radgraph(hyps=predictions, refs=actuals)
        
        print("--- F1radgraph ---")
        print("f1radgraph:", score)
        
        # capture end validation time
        training_time = time.time() - total_t0
        
        # print result summaries
        print("")
        print("==============================================")
        print("Validation Results")
        print("==============================================")
        print("| Epoch | Val loss | ROUGE-L | F1-RadGraph |")
        print(f"| {epoch+1:5d} | {avg_val_loss} | {rougel_f1} | {score} |")
            
    return predictions, actuals, score

def test_generate(epoch, tokenizer, model, device, loader):
    
    # capture validation time
    total_t0 = time.time()
    
    print("")
    print("Running TEST GENETATING...")
    
    model.eval()
    
    total_valid_rouge = 0
    total_valid_loss = 0
    
    predictions = []
    
    pred1 = []
    pred2 = []
    pred3 = []
    
    actuals = []
    val_loss = []
    
    with torch.no_grad():
        for step, batch in enumerate(loader, 0):
            y = batch['target_ids'].to(device, dtype = torch.long)
            ids = batch['source_ids'].to(device, dtype = torch.long)
            mask = batch['source_mask'].to(device, dtype = torch.long)
            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            
            logits = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels, use_cache=False)
            loss = logits[0]
                
            val_loss.append(loss)
            
            ###############################
            start_id = tokenizer.encode('<s>')[0]
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=GENERATION_LEN, 
                num_beams=NUM_BEAMS,
                decoder_start_token_id=start_id,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                num_return_sequences=3,
                # do_sample=True,  # Enable sampling
                )
            ###############################
            
            # Use the tokenizer to convert the output to a string
            # decoded preds    and labels
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            
            # save the generated summaries in separate lists based on their version
            # Reshape the preds list into a nested list with sublists containing num_return_sequences predictions
            # separated_preds = [preds[i:i + num_return_sequences] for i in range(0, len(preds), num_return_sequences)]
            
            batch_size = 4
            num_return_sequences = 3
            
            # Separate predictions based on different versions
            version1, version2, version3 = [], [], []
            for i in range(0, len(preds), num_return_sequences):
                version1.append(preds[i])
                version2.append(preds[i + 1])
                version3.append(preds[i + 2])
                
            v1 = []
            v2 = []
            v3 = []
            
            # Print summaries for each version
            for i in range(batch_size):
#                print(f"Input {i + 1}:")
#                print(f"Version 1: {version1[i]}")
#                print(f"Version 2: {version2[i]}")
#                print(f"Version 3: {version3[i]}")
                v1.append(version1[i])
                v2.append(version2[i])
                v3.append(version3[i])
                
                
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            # rouge_score.add_batch(predictions=preds, references=target)
            # predictions.extend(preds)
            
            actuals.extend(target)
            
            pred1.extend(v1)
            pred2.extend(v2)
            pred3.extend(v3)
            
            if step%200 == 0:
                print(f'Completed {step} step...')
                sys.stdout.flush()
                
        avg_val_loss = statistics.fmean(val_loss)
        print("---validation loss:", avg_val_loss)
        
        # predictions = [str(line) for line in predictions]
        predictions1 = [str(line) for line in pred1]
        predictions2 = [str(line) for line in pred2]
        predictions3 = [str(line) for line in pred3]
        
        actuals = [str(line) for line in actuals]
        
#        print(predictions1)
#        print(predictions2)
#        print(predictions2)
        
        # capture end validation time
        training_time = time.time() - total_t0
        
    return predictions1, predictions2, predictions3, actuals



def main():

    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # Initial Tokenizer and Model 
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    tokenizer.add_tokens(['<s>'])

    model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    print("model", TEXT_MODEL_NAME, model)

    # MIMIC-III original
    # back-translate
    train_dataset = pd.read_json('/home/tongnian/data/mimic-iii/train.json', lines=True)[['findings', 'impression']]
    val_dataset = pd.read_json('/home/tongnian/data/mimic-iii/valid.json', lines=True)[['findings', 'impression']]
    test_dataset = pd.read_json('/home/tongnian/data/mimic-iii/test.json', lines=True)[['findings', 'impression']]
    hidden_test_dataset = pd.read_json('/home/tongnian/data/mimic-iii/hidden_test.json', lines=True)[['findings', 'impression']]

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    print("HIDDEN TEST Dataset: {}".format(hidden_test_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    hidden_test_set = CustomDataset(hidden_test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    
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
        
    # Creation of Dataloaders
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **val_params)
    hidden_test_loader = DataLoader(hidden_test_set, **val_params)

    print("=================")
    print('MODE is: ', MODE)
    
    if MODE == 'train':
        
        exp_folder_path = "./experiments/mimic-iii-output" # using train mode to save model
        print('Start training...', exp_folder_path)
        
        optimizer = torch.optim.AdamW(params=model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps= 1e-08, 
            weight_decay=0.01, # 0. -> small datasets
            amsgrad=True, # False -> small datasets
        )
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

        output_head = ['predict', 'actual']
        with open(exp_folder_path + '/generate_output.csv', 'w', newline='\n') as f:
            dw = csv.DictWriter(f, delimiter=',', fieldnames=output_head)
            dw.writeheader()
        
        # Training loop
        print('Initiating Fine-Tuning...')
        epochs = EPOCHS
        accumulation_steps = GRADIENT_ACCUM
        training_stats = []
        valid_stats = []
        best_val_rouge = 0
        # model.to(device)

        for epoch in range(epochs):

            train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
            sys.stdout.flush()
            
            # validate & generate
            predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
            sys.stdout.flush()
            
            # torch.save(model.state_dict(), model_folder_path + f"/model_ep{epoch}.pt")
                            
            # Save best model based on ROUGE score
            if val_rouge > best_val_rouge:
                best_val_rouge = val_rouge
                
                # save best model for use later
                torch.save(model.state_dict(), exp_folder_path + f"/best_model.pt")
                print(best_val_rouge, ' Best val Rouge Model was saved.')
                
                with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([predictions, actuals])
                    
                # scheduler.step()
                    
                    
    elif MODE == 'test':
        
        print('Start evaluation')
        exp_folder_path = "/home/tongnian/radiologySumm/experiments/clinicalt5-cxr-text-only"
        SAVE_PATH = "/home/tongnian/radiologySumm/test-results/clinicalt5-cxr-text-only"
        
        MODEL_PATH = exp_folder_path + f"/best_model.pt"
        print("MODEL_PATH_SAVE", MODEL_PATH,SAVE_PATH)

        print('EVAL Loading model...')
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(device)
        epoch = 1
        model.eval()
        
        print('Generating TEST output files for evaluation...')
        predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, test_loader)
        sys.stdout.flush()
        process_predictions_and_actuals(predictions, actuals, SAVE_PATH, "output_test")
  
        print('Done.')



if __name__ == '__main__':
    main()
    
    