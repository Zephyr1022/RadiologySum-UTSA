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

    def __init__(self, config_vision,config_text,image_model, text_model):
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
        
        
    def get_embeddings(self, input_ids, attention_mask, pixel_values):
        image_outputs = self.image_model(pixel_values=pixel_values,output_hidden_states=True)
        image_embeds = image_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds.unsqueeze(1) 
        text_embeds = self.text_model.get_input_embeddings()(input_ids) # , attention_mask=attention_mask,output_hidden_states=True
        
        # print("text_embeds shape", text_embeds.shape) # ([2, 512, 768])
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1) # torch.Size([2, 513, 768])
        
        return inputs_embeds, image_embeds
    
    

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        
        inputs_embeds, image_embeds = self.get_embeddings(input_ids,attention_mask,pixel_values)
        
        text_outputs = self.text_model.encoder(inputs_embeds=inputs_embeds) 
        text_embeds = text_outputs.last_hidden_state
        attention_mask = torch.cat([torch.ones((text_embeds.shape[0], 1)).to(device), attention_mask], dim=1)

        # inputs_embeds2 = torch.cat([image_embeds, text_embeds], dim=1)
        # print("att:", inputs_embeds2.shape, attention_mask.shape, torch.ones((inputs_embeds2.shape[0],1)).shape)
        
        # attention_mask = torch.cat([torch.ones((inputs_embeds2.shape[0], 1)).to(device), attention_mask], dim=1)
        # print("mask", attention_mask.shape)

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=text_embeds,
                    hidden_states = text_outputs.hidden_states,
                    attentions=attention_mask,
                )

        



    