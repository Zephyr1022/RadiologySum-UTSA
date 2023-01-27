#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/train.findings.tok') as train_find:
		train_find_list = [line.rstrip('\n') for line in train_find]
	print(len(train_find_list))
	
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/train.impression.tok') as train_imp:
		train_imp_list = [line.rstrip('\n') for line in train_imp]
	print(len(train_imp_list))
	##########################
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/validate.findings.tok') as val_find:
		val_find_list = [line.rstrip('\n') for line in val_find]
	print(len(val_find_list))
	
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/validate.impression.tok') as val_imp:
		val_imp_list = [line.rstrip('\n') for line in val_imp]
	print(len(val_imp_list))
	########################
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/test.findings.tok') as test_find:
		test_find_list = [line.rstrip('\n') for line in test_find]
	print(len(test_find_list))
	
	with open('/home/tongnian/RadiologySumm/data/rrs-mimiciii/all/test.impression.tok') as test_imp:
		test_imp_list = [line.rstrip('\n') for line in test_imp]
	print(len(test_imp_list))
	#########################
	train_dict = {'text':train_find_list, 'summary':train_imp_list} 
	train_df = pd.DataFrame(train_dict) 
	train_df.to_csv('./data/mimiciii-csv/train_data.csv', index=False) 
	
	val_dict = {'text':val_find_list, 'summary':val_imp_list} 
	val_df = pd.DataFrame(val_dict) 
	val_df.to_csv('./data/mimiciii-csv/val_data.csv', index=False) 
	
	test_dict = {'text':test_find_list, 'summary':test_imp_list} 
	test_df = pd.DataFrame(test_dict) 
	test_df.to_csv('./data/mimiciii-csv/test_data.csv', index=False) 
	########################
	read_train = pd.read_csv('./data/mimiciii-csv/train_data.csv')
	print(read_train.shape)
	print(read_train.head())
	
	read_val = pd.read_csv('./data/mimiciii-csv/val_data.csv')
	print(read_val.shape)
	print(read_val.head())
	
	read_test = pd.read_csv('./data/mimiciii-csv/test_data.csv')
	print(read_test.shape)
	print(read_test.head())