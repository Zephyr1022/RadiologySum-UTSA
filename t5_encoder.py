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

    def __init__(self, config_vision,config_text, encoder):
        super().__init__(config_text)
        
        self.config_vision = config_vision
        self.config_text = config_text
        self.text_encoder = encoder
        
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=self.config_vision)
        self.image_model = self.image_model.to(device)
        
        # self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # hidden state
        self.image_embed_dim = self.config_vision.hidden_size # 768
        self.text_embed_dim = self.config_text.hidden_size
        self.projection_dim = self.config_text.hidden_size # the number of output features or dimensions
        
        # dimension projection
        self.visual_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False).to(device)
        # self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False).to(device)
        
        self.dropout = nn.Dropout(0.1)

    def encoder_text_features(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
        
        text_outputs = self.text_encoder(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=return_dict,
                            )
                            
        # pooled_output = text_outputs[1]
        # text_features = self.text_projection(pooled_output)
        
        text_h = text_outputs.last_hidden_state # text_outputs[0]
        text_embeds = text_h # self.text_projection(text_h) 

        return text_embeds
    
    def encoder_image_features(
            self,
            pixel_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        
        vision_outputs = self.image_model(
                        pixel_values=pixel_values,
                        return_dict=return_dict,
                    )
        
        pooled_output = vision_outputs[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)
        
        return image_embeds
    
    
    def get_text_features(self):
        return self.encoder_text_features
    
    def get_image_features(self):
        return self.encoder_image_features

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)


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
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(
            ...     input_ids=inputs.input_ids,
            ...     attention_mask=inputs.attention_mask,
            ...     pixel_values=inputs.pixel_values,
            ... )
        ```"""
        
        # print("t5-encoder", input_ids.shape, attention_mask.shape, pixel_values.shape)

        # Encode image inputs
        # print("PIX:", pixel_values.shape)
        
        image_outputs = self.image_model(pixel_values=pixel_values,output_hidden_states=True)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True) 

        image_embeds = image_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs.last_hidden_state
        # text_embeds = self.text_projection(text_embeds) 

        # concat image and text 
        image_embeds = image_embeds.unsqueeze(1) 
        #print(image_embeds.shape, text_embeds.shape, "SIZES")
        
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        #print("att:", inputs_embeds.shape, attention_mask.shape, torch.ones((text_embeds.shape[0],1)).shape)
        
        attention_mask = torch.cat([torch.ones((text_embeds.shape[0], 1)).to(device), attention_mask], dim=1)
        #print("mask", attention_mask.shape)

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=inputs_embeds,
                    hidden_states = text_outputs.hidden_states,
                    attentions=attention_mask,
                )

        


