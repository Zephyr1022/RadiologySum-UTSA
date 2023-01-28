
import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset
import os, csv, json
from datasets import Dataset

def main():
	
	# Define the name and path of the dataset
	dataset_name = 'text'
	data_dir = './rrs-mimiciii/all'
	
	train_data = {}
	with open(os.path.join(data_dir, 'train.findings.tok')) as f:
		train_data['findings'] = f.readlines()
	with open(os.path.join(data_dir, 'train.impression.tok')) as f:
		train_data['impression'] = f.readlines()
	# Save the train_data dictionary as a json file
	with open('train_data.json', 'w') as f:
		json.dump(train_data, f)
	
	dev_data = {}
	with open(os.path.join(data_dir, 'validate.findings.tok')) as f:
		dev_data['findings'] = f.readlines()
	with open(os.path.join(data_dir, 'validate.impression.tok')) as f:
		dev_data['impression'] = f.readlines()
	# Save the train_data dictionary as a json file
	with open('dev_data.json', 'w') as f:
		json.dump(dev_data, f)
		
		
		
	# Open a new CSV file for writing
#	with open('train_data.csv', 'w', newline='') as csvfile:
#		fieldnames = ['findings', 'impression']
#		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#		writer.writeheader()
#		
#		# Write each row of the train_data dictionary to the CSV file
#		for i in range(len(train_data['findings'])):
#			writer.writerow({'findings': train_data['findings'][i], 'impression': train_data['impression'][i]})
		
			
	
			
	
if __name__ == '__main__':
	main()
	
	
#/home/xingmeng/ViLMedic/rrs-mimiciii/all/train.findings.tok
#/home/xingmeng/ViLMedic/rrs-mimiciii/all/train.impression.tok
#/home/xingmeng/ViLMedic/rrs-mimiciii/all/validate.findings.tok
#/home/xingmeng/ViLMedic/rrs-mimiciii/all/validate.impression.tok