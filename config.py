import os

MODE = 'train'
# Defining some key variables that will be used later on in the training  
GRADIENT_ACCUM = 1
TRAIN_BATCH_SIZE = 1
# this means training will be done for affective batch size of BATCH_SIZE * GRADIENT_ACCUM
EPOCHS = 3
LEARNING_RATE = 1e-3
# random seed
SEED = 42
MAX_LEN = 128
SUMMARY_LEN = 128
MODEL_NAME = 'flan-t5-small'

NUM_BEAMS = 6
NUM_EPOCH = 5
VALID_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
