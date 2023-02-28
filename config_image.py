import os

MODE = 'train'
# Defining some key variables that will be used later on in the training  
GRADIENT_ACCUM = 2
TRAIN_BATCH_SIZE = 1
# this means training will be done for affective batch size of BATCH_SIZE * GRADIENT_ACCUM
EPOCHS = 2
LEARNING_RATE = 1e-2
# random seed
SEED = 123
MAX_LEN = 512
SUMMARY_LEN = 128
MODEL_NAME = 'flan-t5-base'

NUM_BEAMS = 6
NUM_EPOCH = 5
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4