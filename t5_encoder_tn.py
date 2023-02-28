import copy
import torch
import torch.nn as nn
from torch import cuda
from transformers import AutoTokenizer, ViTModel
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
device = 'cuda'


class MultimodalEncoder(T5EncoderModel):

    def __init__(self, configTEXT, configViT):
        super().__init__(configTEXT)

        image_config = copy.deepcopy(configViT)
        encoder_config = copy.deepcopy(configTEXT)

        image_config.output_hidden_states = True
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.output_hidden_states = True
        encoder_config.output_attentions = True

        self.image_embed_dim = image_config.hidden_size # 768
        self.text_embed_dim = encoder_config.hidden_size
        self.projection_dim = encoder_config.hidden_size # the number of output features or dimensions 768
        # Encoder x 2 
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=image_config)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.text_model = T5EncoderModel.from_pretrained('google/flan-t5-base', config=encoder_config)
        self.text_encoder = self.text_model.get_encoder()
        
        # dimension projection
        self.visual_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

    def forward(
        self,
        input_ids = None,
        pixel_values = None,
        attention_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        last_hidden_state = None,
        return_dict = None,
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
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        # Encode text inputs  
        print('ppzi input_ids: ', input_ids.shape) if input_ids is not None else print('ppzi input_ids: ', input_ids)
        print('ppzi attention mask: ', attention_mask.shape)

        text_encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        text_h = text_encoder_outputs.last_hidden_state # text_encoder_outputs[0]
        print("---> text last hidden state: ", text_h.shape) # torch.Size([1, 512, 768])

        text_embeds = self.text_projection(text_h) # (batch_size, sequence_length, projection_dim)
        print("---> text_embeds: ", text_embeds.shape) # torch.Size([1, 512, 768])

        text_hs = text_encoder_outputs.hidden_states
        print("---> text hidden states: ", len(text_hs))

        # Encode image inputs
        image_outputs = self.image_model(pixel_values=pixel_values) # (batch_size, hidden_size)
        image_h = image_outputs[1] # pooled_output
        print("---> image_h: ", image_h.shape) # ([1, 768])

        image_embeds = self.visual_projection(image_h) # (batch_size, projection_dim) 
        image_embeds = image_embeds.unsqueeze(1) # [1, 1, 768]
        print("---> image_embeds: ", image_embeds.shape)

        image_hs = image_outputs.hidden_states
        print("---> image hidden states: ", len(image_hs))

        all_hidden_states = text_hs + image_hs

        if text_embeds.shape[0] == image_embeds.shape[0]:
            # Concat image and text
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1) # [1, 513, 768]
            print("---> inputs_embeds: ", inputs_embeds.shape) # torch.Size([1, 513, 768])
            
            attention_mask = torch.cat([torch.ones((inputs_embeds.shape[0], 1)).to(device), attention_mask], dim=1)
            print("---> attention_mask: ", attention_mask.shape) 
        else:
            repeat_size = text_embeds.shape[0] // image_embeds.shape[0]
            image_embeds = image_embeds.repeat(repeat_size, 1, 1)
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1) # [1, 513, 768]
            print("---> inputs_embeds: ", inputs_embeds.shape) # torch.Size([1, 513, 768])
            
            attention_mask = torch.cat([torch.ones((inputs_embeds.shape[0], 1)).to(device), attention_mask], dim=1)
            print("---> attention_mask: ", attention_mask.shape) 


        return BaseModelOutput(
            last_hidden_state=inputs_embeds, 
            attentions=attention_mask,
            hidden_states = all_hidden_states
            )

        


