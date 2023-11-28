import os

MODE = 'test' # train, inference, embeds

# Defining some key variables that will be used later on in the training  
GRADIENT_ACCUM = 8
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

# this means training will be done for affective batch size of BATCH_SIZE * GRADIENT_ACCUM
EPOCHS = 3
LEARNING_RATE = 1e-4 # 6e-5 # 2e-5 # 1e-4 -> 6e-5

# random seed
SEED = 42
MAX_LEN = 768
SUMMARY_LEN = 128 # max padding len

# Base Model Image
IMAGE_NAME = "google/vit-base-patch16-224"
MODEL_NAME = "google/flan-t5-xl"

# generation
NUM_BEAMS = 2
GENERATION_LEN = 1024

clip_grad = True
max_norm = 1.0
gradient_checkpointing_option = True # True, False
mixed_precision_option = 'no' # no, fp16, bf16
