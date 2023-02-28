# Importing libraries
import os
import logging
import random
import sys
import time
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
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
import optuna 
from optuna.trial import TrialState
from config import *
from torch import cuda
device = 'cuda'

print("test")

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

#Preparing the Dataset for data processing: Class
#defines how the text is pre-processed before sending it to the neural network
#This dataset will be used the the Dataloader method 
#that will feed the data in batches to the neural network for suitable training and processing. 

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
		
#		print("source:", text)
#		print("target:", ctext)
#		print()

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

#def postprocess_text(preds, labels):
#	preds = [pred.strip() for pred in preds]
#	labels = [label.strip() for label in labels]
#	
#	# flatten predictions?
#	all_predictions_flattened = [pred for preds in all_predictions for pred in preds]
#	
#	# ROUGE expects a newline after each sentence
#	preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#	labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
#	
#	return preds, labels

# The dataloader passes data to the model based on the batch size.
# Step 1, Train the model by iterating over all the examples in train_dataloader for each epoch.
# Step 2, Generate model summaries at the end of each epoch, by first generating the tokens and then decoding them (and the reference summaries) into text.
# Step 3, Compute the ROUGE scores using the same techniques we saw earlier.
# Step 4, Save the checkpoints to local. 
	
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
	# capture time
	total_t0 = time.time()
	# Perform one full pass over the training set.
	print("")
	print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
	print('Training...')

	# reset total loss for epoch
	train_total_loss = 0
	total_train_f1 = 0
	
	model.train() # put model into traning mode
	optimizer.zero_grad() # reset gradients for accumulation for the next large_batch
	# for each large batch of training data...
	for idx, batch in enumerate(loader, 0):
		
		# progress update every 40 batches.
		if idx % 200 == 0:# and not idx == 0:
			print(' Batch {:>5,}  of  {:>5,}.'.format(idx, len(loader))) # Report progress.
		
		y = batch['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		ids = batch['source_ids'].to(device, dtype = torch.long)
		mask = batch['source_mask'].to(device, dtype = torch.long)
		
		# outputs = logits
		logits = model(input_ids = ids, 
			attention_mask = mask, 
			decoder_input_ids=y_ids, 
			labels=lm_labels)
		#loss functions
		loss = logits[0]
		
		# sum the training loss over all batches for average loss at end
		# loss is a tensor containing a single value
		train_total_loss += loss.item()
		if idx%200 == 0:
			print({"Training Loss": loss.item()})

		# backpropagation-> gradient accumulation to update the model's parameters
		(loss / accumulation_steps).backward() # gradeints computed for small_batch
		# print('accumu loss:', loss / accumulation_steps)
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
	print("=========================================")
	print("Training Results")
	print("=========================================")
	print("Epoch | train loss | train time ")
	print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")
	# print(f'Epoch: {epoch}, Loss:  {loss.item()}')



# During the validation stage we pass the unseen data(Testing Dataset), trained model, 
# tokenizer and device details to the function to perform the validation run.
# Beam-Search coding
# The generated text and originally summary are decoded from tokens to text
def validate(epoch, epochs, tokenizer, model, device, loader):
	
	# capture validation time
	total_t0 = time.time()
	
	bleu_score = evaluate.load("bleu")
	rouge_score = evaluate.load("rouge")
	f1radgraph = F1RadGraph(reward_level="partial")
	
	# After the completion of each training epoch, measure our performance on
	# our validation set.
	print("")
	print("Running Validation...")
	
	model.eval()
	
	# track variables
	total_valid_rouge = 0
	total_valid_loss = 0
	
	predictions = []
	actuals = []
	
	with torch.no_grad():
		for step, batch in enumerate(loader, 0):
			y = batch['target_ids'].to(device, dtype = torch.long)
			ids = batch['source_ids'].to(device, dtype = torch.long)
			mask = batch['source_mask'].to(device, dtype = torch.long)
			###############################
			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask, 
				max_length=150, 
				num_beams=2,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True
				)
			###############################
			# Use the tokenizer to convert the output to a string
			# decoded preds	and labels
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
			target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
			
			bleu_score.add_batch(predictions=preds, references=target)
			rouge_score.add_batch(predictions=preds, references=target)

			predictions.extend(preds)
			actuals.extend(target)

			if step%200 == 0:
				print(f'Completed {step} step...')
				# print(predictions)
			
		# Compute metrics
		result1 = bleu_score.compute()
		result2 = rouge_score.compute()

		bleu = result1['bleu']
		print("---BLEU:", bleu)
		# Extract the median ROUGE scores
		rouge1_f1 = result2['rouge1']
		rouge2_f1 = result2['rouge2']
		rougel_f1 = result2['rougeL']
		
		print("---ROUGE---")
		print("rouge1:", rouge1_f1)
		print("rouge2:", rouge2_f1)
		print("rougeL:", rougel_f1)
		
		total_valid_rouge = (rouge1_f1+rouge2_f1+rougel_f1)/3
		
#		result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#		result = {k: round(v, 4) for k, v in result.items()}
#		print(f"Epoch {epoch}:", result)
		
		# capture end validation time
		training_time = time.time() - total_t0
		
		# print result summaries
		print("")
		print("=========================================")
		print("Validation Results")
		print("=========================================")
		print("| Epoch | BLEU | ROUGE1 | ROUGE2 | ROUGE-L | Avg Rouge | val time |")
		print(f"| {epoch+1:5d} | {bleu} | {rouge1_f1} | {rouge2_f1} | {rougel_f1} | {total_valid_rouge} | {training_time:} |")
			
	return predictions, actuals, total_valid_rouge


def get_dataloader():
	# tokenzier for encoding the text
	# tokenizer = T5Tokenizer.from_pretrained("t5-base")
	tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
	tokenizer.add_tokens(['<s>'])
	# Importing and Pre-Processing the domain data. Selecting the needed columns only. 
	# Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
	# train_dataset = pd.read_json('./train_data.json', lines=True)[['findings', 'impression']]
	# val_dataset = pd.read_json('./dev_data.json', lines=True)[['findings', 'impression']]
	# test_dataset = pd.read_json('./test_data.json', lines=True)[['findings', 'impression']]
	
	# Modify this line to train data after fine tuning
	train_dataset = pd.read_csv('./data/mimiciii-csv/val_data.csv',nrows=200)
	# ------------------------------------------------
	val_dataset = pd.read_csv('./data/mimiciii-csv/val_data.csv',nrows=200)
	test_dataset = pd.read_csv('./data/mimiciii-csv/test_data.csv')
	
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

	return training_loader, val_loader, test_loader, tokenizer


def objective(trial):

	# Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
	# model.resize_token_embeddings(len(tokenizer))
	# model.half()
	# model = T5ForConditionalGeneration.from_pretrained("t5-base")
	model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
	model = model.to(device)

	epochs = trial.suggest_int("epochs", low=8, high=12)
    # Generate the optimizers
	optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
	lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
	
	# Get the dataloader of the dataset
	training_loader, val_loader, test_loader, tokenizer = get_dataloader()

	# write the generated output to a csv file
	output_head = ['predict', 'actual']
	with open('./experiments/flan-t5-base/hyperparameter/generate_output.csv', 'w', newline='\n') as f:
		dw = csv.DictWriter(f, delimiter=',', fieldnames=output_head)
		dw.writeheader()
	
	# Training and validation process
	print('Initiating Fine-Tuning...')
	# epochs = EPOCHS
	accumulation_steps = GRADIENT_ACCUM
	training_stats = []
	valid_stats = []
	best_val_rouge = 0

	for epoch in range(epochs):
		model.to(device)
		# training
		train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
		sys.stdout.flush()
		
		# validate & generate
		predictions, actuals, val_rouge = validate(epoch, epochs, tokenizer, model, device, val_loader)
		sys.stdout.flush()

		trial.report(val_rouge, epoch)
		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

		# save model per epoch
		torch.save(model.state_dict(), f"./model_save/debug/model_ep{epoch}.pt")
		# save generated output per epoch
		with open('./model_save/debug/generate_output.csv', 'a', newline='\n') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow([predictions, actuals])
		
		# Save best model based on ROUGE score
		if val_rouge > best_val_rouge:
			best_val_rouge = val_rouge
			# predictions_test, actuals_test, test_rouge = validate(epoch,epochs, tokenizer, model, device, test_loader)
			torch.save(model.state_dict(), "./model_save/debug/best_model.pt")
			print(best_val_rouge, ' Best val Rouge Model was saved.')
		# scheduler.step()
	return val_rouge


def main():
	
	# # Defining some key variables that will be used later on in the training  
	# GRADIENT_ACCUM = 4
	# TRAIN_BATCH_SIZE = 4
	# # this means training will be done for affective batch size of BATCH_SIZE * GRADIENT_ACCUM
	# VALID_BATCH_SIZE = 4
	# TEST_BATCH_SIZE = 16
	# EPOCHS = 8
	# LEARNING_RATE = 1e-2
	# # random seed
	# SEED = 123
	# MAX_LEN = 512
	# SUMMARY_LEN = 128
	# MODEL_NAME = 'flan-t5-base'
	
	# Set random seeds and deterministic pytorch for reproducibility
	torch.manual_seed(SEED) # pytorch random seed
	np.random.seed(SEED) # numpy random seed
	torch.backends.cudnn.deterministic = True
	# # creating experiment folders
	# exp_folder_path = create_folder(f"./experiments/{MODEL_NAME}/hyperparameter", f"ep_{EPOCHS}_batch_{TRAIN_BATCH_SIZE}_step_{GRADIENT_ACCUM}_lr_{LEARNING_RATE}")
	# model_folder_path = create_folder(exp_folder_path, "model_save")
	# # configuring and saving Logger & Output
	# Log_Format = "%(levelname)s | %(asctime)s | %(filename)s | %(message)s"
	# logging.basicConfig(filename = exp_folder_path + "/logger.log",
    #                     filemode = "a",
    #                     format = Log_Format, 
    #                     level = logging.DEBUG)
	# logger = logging.getLogger()
	# log_file = open(exp_folder_path + "/output.log","w")
	# sys.stdout = log_file

	print(" Start a tunning process")
	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=2)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("===Study statistics===")
	print("   Number of finished trials: ", len(study.trials))
	print("   Number of pruned trials: ", len(pruned_trials))
	print("   Number of complete trials: ", len(complete_trials))

	print("===Best trial:")
	trial = study.best_trial
	print("   Best Value: ", trial.value)
	print("   Best Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))



	# Defining the optimizer that will be used to tune the weights of the network in the training session. 
	# optimizer = torch.optim.SGD(params =  model.parameters(), lr=LEARNING_RATE)
	# optimizer = Adafactor(params =  model.parameters(), lr=LEARNING_RATE, scale_parameter=False, relative_step=False)
	# optimizer = torch.optim.AdamW(params=model.parameters(),
	# 	lr=LEARNING_RATE,
	# 	betas=(0.9, 0.999),
	# 	eps= 1e-08, 
	# 	weight_decay=0.,
	# 	amsgrad=False)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
	#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

	# output_head = ['predict', 'actual']
	# with open(exp_folder_path + '/generate_output.csv', 'w', newline='\n') as f:
	# 	dw = csv.DictWriter(f, delimiter=',', fieldnames=output_head)
	# 	dw.writeheader()
	
	# # Training loop
	# print('Initiating Fine-Tuning...')
	# epochs = EPOCHS
	# accumulation_steps = GRADIENT_ACCUM
	# training_stats = []
	# valid_stats = []
	# best_val_rouge = 0

	# for epoch in range(epochs):
	# 	model.to(device)
		
	# 	train(epoch, epochs, tokenizer, model, device, training_loader, optimizer, accumulation_steps)
	# 	sys.stdout.flush()
		
	# 	# validate & generate
	# 	predictions, actuals, val_rouge = validate(epoch, epochs, tokenizer, model, device, val_loader)
	# 	sys.stdout.flush()
		
	# 	torch.save(model.state_dict(), model_folder_path + f"/model_ep{epoch}.pt")
		
	# 	with open(exp_folder_path + '/generate_output.csv', 'a', newline='\n') as f:
	# 		writer = csv.writer(f, delimiter=',')
	# 		writer.writerow([predictions, actuals])
		
	# 	# Save best model based on ROUGE score
	# 	if val_rouge > best_val_rouge:
	# 		best_val_rouge = val_rouge
	# 		# predictions_test, actuals_test, test_rouge = validate(epoch,epochs, tokenizer, model, device, test_loader)
			
	# 		# save best model for use later
	# 		torch.save(model.state_dict(), model_folder_path + f"/best_model.pt")
	# 		print(best_val_rouge, ' Best val Rouge Model was saved.')
	# 	# scheduler.step()

#Evaluate the model performance with the sumeval's Rouge score.
#Rouge1: Evaluate the generated text in units of uni-grams.
#Rouge2: Evaluate the generated text in units of bi-grams.
#RougeL: Evaluate the match of the generated text sequence.
	
if __name__ == '__main__':
	main()