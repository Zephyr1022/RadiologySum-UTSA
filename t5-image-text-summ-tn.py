import os
import logging
import random
import copy
import sys
import time
import csv
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
# import optuna
from PIL import Image
from transformers import BertConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import ViTConfig, ViTModel, AutoImageProcessor
from transformers import AutoConfig, AutoFeatureExtractor
import inspect
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
from t5_encoder_tn import MultimodalEncoder
from transformers import GenerationConfig
from config import *
from torch import cuda
device = 'cuda'


# The main idea is to expand the T5ForConditionalGeneration class to handle the image embeddings.
class MultiModalt5(T5ForConditionalGeneration):  # nn.Module
    def __init__(self, configTEXT, configViT):
        super().__init__(configTEXT) # MultiModalt5, self
        
        self.configTEXT = configTEXT
        self.configViT = configViT
        
        self.image_embed_dim = configViT.hidden_size # 768
        self.text_embed_dim = configTEXT.hidden_size
        self.projection_dim = configTEXT.hidden_size # the number of output features or dimensions
        # Encoder x 2 
        # ANTHONY: You don't need to initialize T5ForConditionalGeneration() here, you may be able to use T5Stack to load the decoder directly.
        # ANTHONY: See the T5ForConditionalGeneration() class here https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_t5.html
        # self.text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', config=self.configTEXT) 
        # self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=self.configViT)
        self.configTEXT.use_cache = True
        self.shared = nn.Embedding(self.configTEXT.vocab_size, self.configTEXT.d_model)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.encoder = MultimodalEncoder(self.configTEXT, self.configViT)
        # ANTHONY: If you use the T5Stack class as mentioned above, you don't need to use get_decoder().
        decoder_config = copy.deepcopy(self.configTEXT)

        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.configTEXT.num_decoder_layers
        

        self.text_decoder = T5Stack(decoder_config, self.shared)
        # ANTHONY: It would be nice to load lm_head without having duplicate encoder parameters loaded on the GPU. Not sure how to handle that part though.

        # Decoder
        self.visual_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # self.enc_to_dec_proj = nn.Linear(709, self.projection_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    # ANTHONY: You should add a get_encoder() and get_decoder() method here.
    # ANTHONY: See https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_t5.html with an example in the T5ForConditionalGeneration class.

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, decoder_input_ids=None, labels=None, encoder_outputs=None, use_cache=True, **kwargs):
        use_cache = use_cache if use_cache is not None else self.configTEXT.use_cache
        # Encoder output
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        # # text output from encoder
        # text_h = encoder_outputs.last_hidden_state # text_encoder_outputs[0]
        # print("---> text_h", text_h.shape) # torch.Size([1, 512, 768])
        # text_embeds = self.text_projection(text_h) # (batch_size, sequence_length, projection_dim)
        # print("---> text_embeds", text_embeds.shape) # torch.Size([1, 512, 768])

        # image output from encoder
        # image_outputs = self.image_model(pixel_values=pixel_values) # (batch_size, hidden_size)
        # image_h = image_outputs[1] # pooled_output
        # print("---> image_h:", image_h.shape) # ([1, 768])
        # image_embeds = self.visual_projection(image_h) # (batch_size, projection_dim) 
        # image_embeds = image_embeds.unsqueeze(1) # [1, 1, 768]
        # print("---> image_embeds", image_embeds.shape)

        # Concat image and text
        # inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1) # [1, 513, 768]
        # print("---> inputs_embeds", inputs_embeds.shape) # torch.Size([1, 513, 768])
        # encoder_outputs.last_hidden_state = inputs_embeds
        # attention_mask = torch.cat([torch.ones((1, 1)).to(device), attention_mask], dim=1)
        # print("attention_mask", attention_mask.shape) 
        # encoder_outputs.attentions = attention_mask
        # Decode the concatenated hidden states using the T5 decoder -> (batch_size, sequence_length, hidden_size)

        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_outputs.last_hidden_state, # Last hidden states combined with image embeds
                        encoder_attention_mask=encoder_outputs.attentions,
                        use_cache=use_cache,
                        )
        
        last_linear_layer = self.lm_head
        logits = last_linear_layer(decoder_outputs[0])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
                    loss=loss,
                    logits=logits,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state, #encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )
                
    def prepare_inputs_for_generation(
        self,
        input_ids,
        decoder_input_ids = None,
        pixel_values=None,
        encoder_outputs=None,
        encoder_hidden_states=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # attention_mask = input_ids.new_ones(input_ids.shape)
        # input_ids = input_ids.expand(-1, attention_mask.shape[1])
        print("ppzi prepare inputs for generation:", input_ids.shape, attention_mask.shape, pixel_values.shape)
        # encoder_outputs = self.encoder(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        # encoder_hidden_states = encoder_outputs.hidden_states
        # print('ppzi prepare inputs for generation encoder hidden states: ', len(encoder_hidden_states))
        return {
            "decoder_input_ids": input_ids,
            "pixel_values": pixel_values,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    


class CustomDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, image_processor, source_len, summ_len):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.findings
        self.ctext = self.data.impression
        self.image = self.data.image
        #self.text = self.data['text'] # This is the complete findings 
        #self.ctext = self.data['summary'] # This is the summary of the findings
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        
        """
            Constructs an image processor and a tokenizer into a single processor.
        """
        
        text = str(self.text[index])
        text = ' '.join(text.split())
        
        ctext = str(self.ctext[index])
        ctext = '<s> '+' '.join(ctext.split()) # df.ctext = 'summarize: ' + df.ctext/ inputs = ["summarize: " + text]
        
        # multiple image??? ~ one summary
        image_path = str(self.image[index])
        image_path ='/home/tongnian/data/mimic-cxr/'+ ''.join(image_path.split(',')[0])
        
        # error handling
        try:
            ct_image = Image.open(image_path).convert('RGB') #.convert('RGB')
            # print('ct_image shape:', ct_image.shape)
        except:
            print(f"Error opening image file {image_path}")
            return None

        # model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        # image_features = model.get_image_features(**inputs)
        # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py

        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len, padding='max_length', return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len, padding='max_length', return_tensors='pt',truncation=True)
        image_features = self.image_processor(ct_image, return_tensors="pt")  # do we need to padding? 
        
        print(image_features.pixel_values.shape)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()        
        
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'pixel_values': image_features.pixel_values #pixel_values:[1,3,224,224]
    }
    
    
def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
    total_t0 = time.time()
    print('Training...')

    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode
    optimizer.zero_grad() # reset gradients for accumulation for the next large_batch
    # for each large batch of training data...
    for idx, batch in enumerate(loader, 0):
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long)
        image = batch['pixel_values'].to(device)
        image = torch.squeeze(image, dim=1) 
        print('==========', y_ids.shape)
        # outputs = logits, loss function
        logits = model(input_ids = ids, attention_mask = mask, pixel_values = image, decoder_input_ids=y_ids, labels=lm_labels)
        loss = logits[0]
        
        print("logits, loss:",loss)
        
        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()
        if idx%100 == 0:
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
            image = batch['pixel_values'].to(device)
            image = torch.squeeze(image, dim=1)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            print('ppzi validate: ', ids.shape, mask.shape, y_ids.shape)
            logits = model(input_ids = ids, attention_mask = mask, pixel_values = image, decoder_input_ids=y_ids, labels=lm_labels)
            loss = logits[0]
            val_loss.append(loss)
            start_id = tokenizer.encode('<s>')[0]
            ###############################
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=NUM_BEAMS,
                decoder_start_token_id=start_id,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                pixel_values=image,
                )
            ###############################
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            rouge_score.add_batch(predictions=preds, references=target)
            predictions.extend(preds)
            actuals.extend(target)
            if step%100 == 0:
                print(f'Completed {step} step...')
        avg_val_loss = statistics.fmean(val_loss)
        print("---validation loss:", avg_val_loss)
        result2 = rouge_score.compute()
        rouge1_f1 = result2['rouge1']
        rouge2_f1 = result2['rouge2']
        rougel_f1 = result2['rougeL']
        print("--- ROUGE ---")
        print("rouge1:", rouge1_f1)
        print("rouge2:", rouge2_f1)
        print("rougeL:", rougel_f1)
        total_valid_rouge = (rouge1_f1+rouge2_f1+rougel_f1)/3
        print("")
        print("==============================================")
        print("Validation Results")
        print("==============================================")
        print("| Epoch | Val loss | ROUGE1 | ROUGE2 | ROUGE-L | Avg Rouge |")
        print(f"| {epoch+1:5d} | {avg_val_loss} | {rouge1_f1} | {rouge2_f1} | {rougel_f1} | {total_valid_rouge} |")
    return predictions, actuals, total_valid_rouge


def main():
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.add_tokens(['<s>'])
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    config_text = AutoConfig.from_pretrained("google/flan-t5-base")
    config_vision = ViTConfig()

    model = MultiModalt5(configTEXT = config_text, configViT = config_vision)
    model = model.to(device)
    
    # print("model", model)

    #model = T5EncoderModel.from_pretrained("google/flan-t5-base")
    #model = model.to(device)
    #model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    #model = model.to(device)

    # Importing and Pre-Processing the domain data. Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    train_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/train.json', lines=True)[['findings', 'impression','image']].iloc[:30]
    val_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/valid.json', lines=True)[['findings', 'impression','image']].iloc[:20]
    test_dataset = pd.read_json('/home/tongnian/data/mimic-cxr/test.json', lines=True)[['findings', 'impression','image']].iloc[:20]

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)
    test_set = CustomDataset(test_dataset, tokenizer,image_processor, MAX_LEN, SUMMARY_LEN)

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
        
    # Creation of Dataloaders # batch
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **val_params)

    print("training_set", training_loader)
    
    
    print("=================")
    print('MODE is: ', MODE)
    
    if MODE == 'train' or MODE == 'test':
        print('Start training')
        
        optimizer = torch.optim.AdamW(params=model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps= 1e-08, 
            weight_decay=0.,
            amsgrad=False)
        
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

            predictions, actuals, val_rouge = validate(epoch, tokenizer, model, device, val_loader)
            sys.stdout.flush()

#Evaluate the model performance with the sumeval's Rouge score.
#Rouge1: Evaluate the generated text in units of uni-grams.
#Rouge2: Evaluate the generated text in units of bi-grams.
#RougeL: Evaluate the match of the generated text sequence.
    
if __name__ == '__main__':
    main()

    
