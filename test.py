class BabyModel(GPT2LMHeadModel): # nn.Module
    def __init__(self, num_labels):
        
        # config = AutoConfig.from_pretrained(checkpoint)
        # config = GPT2Config.from_pretrained("gpt2") 
        config = BabyConfig.from_pretrained("custom-babylm")
        config.output_attentions = True
        config.output_hidden_states = True

        super(BabyModel, self).__init__(config) #GPT2LMHeadModel(config)
        self.num_labels = num_labels
        
        # Load Model with given checkpoint and extract its body
        self.model = T5EncoderModel.from_pretrained("flan-t5", config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) # initialize weights

    def forward(self, input_ids=None, output_ids, image=None, attention_mask=None, labels=None):
        # extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) 

        last_hidden_states = outputs.last_hidden_state

        image_output = self.image_model(image)

        image_hidden = image_output.last_hidden_state

        h = concat([last_hidden_states, image_hidden])

        new_vec = relu(self.classifier(h))


        outputs = self.model_decoder(new_vec, output_ids)

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    

