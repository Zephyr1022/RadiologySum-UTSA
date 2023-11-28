import torch
import torch.nn as nn
from torch import cuda
from transformers import AutoTokenizer, ViTModel
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.modeling_utils import PreTrainedModel

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack, T5PreTrainedModel
from transformers import AutoConfig
from transformers import ViTConfig, ViTModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

device = 'cuda'

class MultimodalEncoder(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config_vision,config_text,image_model,text_model):
        super().__init__(config_text)
        
        self.config_vision = config_vision
        self.config_text = config_text
        
        self.text_model = text_model
        self.image_model = image_model

        # hidden state
        self.image_embed_dim = self.config_vision.hidden_size
        self.text_embed_dim = self.config_text.hidden_size
        self.projection_dim = self.config_text.hidden_size
        
        # dimension projection
        self.visual_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False).to(device)
        self.dropout = nn.Dropout(0.1)
        
    # Input embeddings are the initial representations of the tokens
    # and then, take the last_hidden_state from the T5 encoder output as embeddings
    # self.text_model.get_input_embeddings()(input_ids)
    def get_text_encoder_embeds(self, input_ids, attention_mask):
        
        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict = True,
        )
        
        batch_text_embeds = encoder_outputs.last_hidden_state
        
        # Extract the last token embeddings for each sequence in the batch
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_embeds = batch_text_embeds[torch.arange(batch_text_embeds.size(0)), last_token_indices] # EOS token
        
        return last_token_embeds
    
    
    def get_text_image_encoder_embeds(self, input_ids, attention_mask,pixel_values, num_images):
        
        image_embeds, image_atts = self.get_embeddings(pixel_values,num_images)
        
        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_atts, 
            output_hidden_states=True,
            return_dict = True,
        )
        
        batch_text_embeds = encoder_outputs.last_hidden_state
        
        # Extract the last token embeddings for each sequence in the batch
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_embeds = batch_text_embeds[torch.arange(batch_text_embeds.size(0)), last_token_indices] # EOS token for encoder-decoder model like [CLS] token in bert 
        
        return last_token_embeds
    
        
    def get_embeddings(self, pixel_values, num_images):
        
        # input_ids, pixel_values: torch.Size([2, 512]) torch.Size([2, 3, 3, 224, 224])
        # print("input_ids", input_ids.shape, pixel_values.shape, "batch size", len(num_images), num_images) 
        batch_size, max_num_images, image_color, image_x, image_y = pixel_values.size()
        pixel_values = pixel_values.view(batch_size * max_num_images, image_color, image_x, image_y)

        batch_image_feature = []
        for idx, num_image in enumerate(num_images):
            
            image_features = []
            idx_new = idx * max_num_images # 0*3
            
            for image_ in range(num_image):
                
                p = pixel_values[image_+idx_new]
                # print("num_image",image_+idx_new, num_image, p.shape)
                
                image_outputs = self.image_model(pixel_values=p.unsqueeze(0), output_hidden_states=True)
                image_embeds = image_outputs[1]
                image_embeds = self.visual_projection(image_embeds) # do we need this projection??? The projection should be trained for sure, Just the weights to the model itself.
                
                image_features.append(image_embeds)
                
            average_image_features = torch.mean(torch.stack(image_features), dim=0)
            batch_image_feature.append(average_image_features)
            
        batch_image_embeds = torch.stack(batch_image_feature, dim=0) # torch.Size([2, 1, 768])
        image_atts = torch.ones(batch_image_embeds.size()[:-1],dtype=torch.long).to(device)
        
        return batch_image_embeds, image_atts

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        num_images=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        image_embeds, image_atts = self.get_embeddings(pixel_values,num_images)
        
        text_outputs = self.text_model.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_atts, 
            output_hidden_states=True,
            return_dict = True,
        )

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state = text_outputs.last_hidden_state,
                    hidden_states = text_outputs.hidden_states,
                    attentions = attention_mask,
                )
    
    
'''
https://huggingface.co/course/chapter7/5

option1
text -> encoder -> encoding text + image -> decoder

option 2
text -> token embeddings -> token embeddings + image embedding -> encoder -> decoder
    
option 3
text -> token embeddings -> token embeddings + image embedding -> encoder -> encoder embeddings + original image embedding -> decoder
'''