# Importing libraries
import random
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import Adafactor, AdafactorSchedule

import transformers
from datasets import load_dataset, load_metric

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Setting up the device for GPU usage
from torch import cuda

device = 'cuda'

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
		self.text = self.data.findings # This is the complete article 
		self.ctext = self.data.impression # This is the summary of the article
		
	def __len__(self):
		return len(self.text)
	
	def __getitem__(self, index):
		ctext = str(self.ctext[index])
		ctext = '<s> '+' '.join(ctext.split())
		
		text = str(self.text[index])
		text = ' '.join(text.split())
		
		source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt',truncation=True)
		target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt',truncation=True)
		
		print("source:", text)
		print("target:", ctext)
		print()

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


# The dataloader passes data to the model based on the batch size.
def train(epoch, tokenizer, model, device, loader, optimizer):
	model.train()
	for _,data in enumerate(loader, 0):
		y = data['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		ids = data['source_ids'].to(device, dtype = torch.long)
		mask = data['source_mask'].to(device, dtype = torch.long)
		
		outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
		loss = outputs[0]


		print({"Training Loss": loss.item()})
		
		if _%500==0:
			print(f'Epoch: {epoch}, Loss:  {loss.item()}')
			
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# xm.optimizer_step(optimizer)
		# xm.mark_step()

# During the validation stage we pass the unseen data(Testing Dataset), trained model, 
# tokenizer and device details to the function to perform the validation run.
# Beam-Search coding
# The generated text and originally summary are decoded from tokens to text
def validate(epoch, tokenizer, model, device, loader):
	model.eval()
	predictions = []
	actuals = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			y = data['target_ids'].to(device, dtype = torch.long)
			ids = data['source_ids'].to(device, dtype = torch.long)
			mask = data['source_mask'].to(device, dtype = torch.long)
			
			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask, 
				max_length=150, 
				num_beams=2,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True
				)
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
			target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
			if _%100==0:
				print(f'Completed {_}')
				
			predictions.extend(preds)
			actuals.extend(target)
	return predictions, actuals

def main():
	
	# Defining some key variables that will be used later on in the training  
	TRAIN_BATCH_SIZE = 2   # input batch size for training (default: 64)
	VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
	TRAIN_EPOCHS = 2       # number of epochs to train (default: 10)
	VAL_EPOCHS = 2 
	LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
	SEED = 123               # random seed (default: 42)
	MAX_LEN = 512
	SUMMARY_LEN = 128
	
	# Set random seeds and deterministic pytorch for reproducibility
	torch.manual_seed(SEED) # pytorch random seed
	np.random.seed(SEED) # numpy random seed
	torch.backends.cudnn.deterministic = True

	# tokenzier for encoding the text
	tokenizer = T5Tokenizer.from_pretrained("t5-base")
	tokenizer.add_tokens(['<s>'])
	
	# Importing and Pre-Processing the domain data
	# Selecting the needed columns only. 
	# Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 

	train_dataset = pd.read_json('./train_data.json', lines=True)[['findings', 'impression']]
	val_dataset = pd.read_json('./dev_data.json', lines=True)[['findings', 'impression']]
	
	print(train_dataset.head())

	print("TRAIN Dataset: {}".format(train_dataset.shape))
	print("TEST Dataset: {}".format(val_dataset.shape))
	
	# Creating the Training and Validation dataset for further creation of Dataloader
	training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
	val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

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
		
	# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
	training_loader = DataLoader(training_set, **train_params)
	val_loader = DataLoader(val_set, **val_params)
	
	# Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
	# Further this model is sent to device (GPU/TPU) for using the hardware.
	model = T5ForConditionalGeneration.from_pretrained("t5-base")
	# model.resize_token_embeddings(len(tokenizer))
	model = model.to(device)
	# model.half()
	
	# Defining the optimizer that will be used to tune the weights of the network in the training session. 
	# optimizer = torch.optim.SGD(params =  model.parameters(), lr=LEARNING_RATE)
	# optimizer = Adafactor(params =  model.parameters(), lr=LEARNING_RATE, scale_parameter=False, relative_step=False)
	optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
	
	# Training loop
	print('Initiating Fine-Tuning for the model on our dataset')
	
	for epoch in range(TRAIN_EPOCHS):
		train(epoch, tokenizer, model, device, training_loader, optimizer)
	
	# Validation loop and saving the resulting file with predictions and acutals in a dataframe.
	# Saving the dataframe as predictions.csv
	
	torch.save(model.state_dict(), './t5_model_schema_largsos_token2.pt')
	
	
#	checkpoint = torch.load('./t5_large.pt') # Load fine-tuned model
#	model.load_state_dict(checkpoint) #['model_state_dict'])
#	print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
#	for epoch in range(VAL_EPOCHS):
#		predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
#		final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
#		final_df.to_csv('./models/predictions.csv')
#		print('Output Files generated for review')

	
if __name__ == '__main__':
	main()