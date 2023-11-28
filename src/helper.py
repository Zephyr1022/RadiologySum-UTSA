# Helper Functions
import os 

def postprocess_text(preds, labels):
	
	preds = [pred.strip() for pred in preds]
	labels = [label.strip() for label in labels]
	
	# ROUGE expects a newline after each sentence
	preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
	labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
	
	return preds, labels