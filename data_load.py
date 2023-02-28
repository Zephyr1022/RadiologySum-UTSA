import pandas as pd


def main():

    with open('./data/mimic-cxr/train.findings.tok') as f:
        train_find_list = [line.rstrip('\n') for line in f]
    print(len(train_find_list))
    
    with open('./data/mimic-cxr/train.impression.tok') as f:
        train_imp_list = [line.rstrip('\n') for line in f]
    print(len(train_imp_list))

    with open('./data/mimic-cxr/train.image.tok') as f:
        train_img_list = [line.rstrip('\n') for line in f]
    print(len(train_img_list))

    with open('./data/mimic-cxr/validate.findings.tok') as f:
        val_find_list = [line.rstrip('\n') for line in f]
    print(len(val_find_list))

    with open('./data/mimic-cxr/validate.impression.tok') as f:
        val_imp_list = [line.rstrip('\n') for line in f]
    print(len(val_imp_list))

    with open('./data/mimic-cxr/validate.image.tok') as f:
        val_img_list = [line.rstrip('\n') for line in f]
    print(len(val_img_list))

    with open('./data/mimic-cxr/test.findings.tok') as f:
        test_find_list = [line.rstrip('\n') for line in f]
    print(len(test_find_list))

    with open('./data/mimic-cxr/test.impression.tok') as f:
        test_imp_list = [line.rstrip('\n') for line in f]
    print(len(test_imp_list))

    with open('./data/mimic-cxr/test.image.tok') as f:
        test_img_list = [line.rstrip('\n') for line in f]
    print(len(test_img_list))

    train_dict = {'text':train_find_list, 'summary':train_imp_list, 'image_dir': train_img_list} 
    train_df = pd.DataFrame(train_dict) 
    train_df.to_csv('./data/mimic-cxr/train_data.csv', index=False) 
    
    val_dict = {'text':val_find_list, 'summary':val_imp_list, 'image_dir': val_img_list} 
    val_df = pd.DataFrame(val_dict) 
    val_df.to_csv('./data/mimic-cxr/val_data.csv', index=False) 

    test_dict = {'text':test_find_list, 'summary':test_imp_list, 'image_dir': test_img_list} 
    test_df = pd.DataFrame(test_dict) 
    test_df.to_csv('./data/mimic-cxr/test_data.csv', index=False) 

    read_train = pd.read_csv('./data/mimic-cxr/train_data.csv')
    print(read_train.shape)
    print(read_train.head())

    read_val = pd.read_csv('./data/mimic-cxr/val_data.csv')
    print(read_val.shape)
    print(read_val.head())

    read_test = pd.read_csv('./data/mimic-cxr/test_data.csv')
    print(read_test.shape)
    print(read_test.head())


if __name__ == '__main__':
	main()