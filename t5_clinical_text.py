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
# from radgraph import F1RadGraph
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import optuna 
from config import *
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


class CustomDataset(Dataset):
	
	def __init__(self, dataframe, tokenizer, source_len, summ_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.source_len = source_len
		self.summ_len = summ_len
		self.text = self.data['text'] # This is the complete findings 
		self.ctext = self.data['summary'] # This is the summary of the findings
		
	def __len__(self):
		return len(self.text)
	
	def __getitem__(self, index):
		
		text = str(self.text[index])
		text = ' '.join(text.split())
		
		ctext = str(self.ctext[index])
		ctext = '<s> '+' '.join(ctext.split()) # df.ctext = 'summarize: ' + df.ctext/ inputs = ["summarize: " + text]

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
			'target_ids_y': target_ids.to(dtype=torch.long)
		}
	
	
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
	# capture time
	total_t0 = time.time()
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
				max_length=150, 
				num_beams=NUM_BEAMS,
				decoder_start_token_id=start_id,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True
				)
			###############################
			
			# Use the tokenizer to convert the output to a string
			# decoded preds	and labels
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
		# result3 = bertscore.compute(lang="en")

		#bleu = result1['bleu']
		#print("--- BLEU ---:", bleu)
		rouge1_f1 = result2['rouge1']
		rouge2_f1 = result2['rouge2']
		rougel_f1 = result2['rougeL']
		
		print("--- ROUGE ---")
		print("rouge1:", rouge1_f1)
		print("rouge2:", rouge2_f1)
		print("rougeL:", rougel_f1)
		
		ave_valid_rouge = (rouge1_f1+rouge2_f1+rougel_f1)/3
		
		# capture end validation time
		training_time = time.time() - total_t0
		
		# print result summaries
		print("")
		print("==============================================")
		print("Validation Results")
		print("==============================================")
		print("| Epoch | Val loss | ROUGE1 | ROUGE2 | ROUGE-L | Avg Rouge |")
		print(f"| {epoch+1:5d} | {avg_val_loss} | {rouge1_f1} | {rouge2_f1} | {rougel_f1} | {ave_valid_rouge} |")
			
	return predictions, actuals, rougel_f1



def main():

	torch.manual_seed(SEED) # pytorch random seed
	np.random.seed(SEED) # numpy random seed
	torch.backends.cudnn.deterministic = True
	
	# creating experiment folders
	exp_folder_path = create_folder(f"./experiments/clinicalt5-cxr-text-only", f"ep_{EPOCHS}_batch_{TRAIN_BATCH_SIZE}_step_{GRADIENT_ACCUM}_lr_{LEARNING_RATE}")
	model_folder_path = create_folder(exp_folder_path, "model_save")

	tokenizer = AutoTokenizer.from_pretrained("/home/tongnian/RadiologySumm/clinical_t5/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Large")
	tokenizer.add_tokens(['<s>'])
	
	model = AutoModelForSeq2SeqLM.from_pretrained("/home/tongnian/RadiologySumm/clinical_t5/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Large")
	model = model.to(device)
	
	# Importing and Pre-Processing the domain data. Selecting the needed columns only. 
	# Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
	# train_dataset = pd.read_json('./train_data.json', lines=True)[['findings', 'impression']]
	# val_dataset = pd.read_json('./dev_data.json', lines=True)[['findings', 'impression']]
	# test_dataset = pd.read_json('./test_data.json', lines=True)[['findings', 'impression']]

#	mimiciii-csv/  # MIMIC-III
#	mimic-cxr/ # MIMIC-CXR and CheXpert

	train_dataset = pd.read_csv('/home/tongnian/data/mimic-cxr/train_data.csv')
	val_dataset = pd.read_csv('/home/tongnian/data/mimic-cxr/val_data.csv')
	test_dataset = pd.read_csv('/home/tongnian/data/mimic-cxr/test_data.csv')
	
	print("TRAIN Dataset: {}".format(train_dataset.shape))
	print("VALID Dataset: {}".format(val_dataset.shape))
	print("TEST Dataset: {}".format(test_dataset.shape))
	
	# Creating the Training and Validation dataset for further creation of Dataloader
	training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
	val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
	test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

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

	print("=================")
	print('MODE is: ', MODE)
	
	if MODE == 'train':
		
		log_file = open(exp_folder_path + "/output.log","w")
		sys.stdout = log_file
		print('Start training...')
		
		optimizer = torch.optim.AdamW(params=model.parameters(),
			lr=LEARNING_RATE,
			betas=(0.9, 0.999),
			eps= 1e-08, 
			weight_decay=0.,
			amsgrad=False)
		
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

		for epoch in range(epochs):
			model.to(device)
			train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
			sys.stdout.flush()
			# validate & generate
			predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
			sys.stdout.flush()
			# torch.save(model.state_dict(), model_folder_path + f"/model_ep{epoch}.pt")
			
			with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
				writer = csv.writer(f, delimiter=',')
				writer.writerow([predictions, actuals])
			
			# Save best model based on ROUGE score
			if val_rouge > best_val_rouge:
				best_val_rouge = val_rouge
				# save best model for use later
				torch.save(model.state_dict(), model_folder_path + f"/best_model.pt")
				print(best_val_rouge, ' Best val Rouge Model was saved.')
			# scheduler.step()


if __name__ == '__main__':
	main()
	
