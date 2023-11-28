import statistics
import time
import torch
import evaluate

def train(epoch, epochs, tokenizer, model, device, loader, optimizer, accumulation_steps):
    
    total_t0 = time.time()
    train_total_loss = 0
    total_train_f1 = 0
    
    model.train() # put model into traning mode

    # each item in batch?
    for idx, batch in enumerate(loader, 0):
        
        y = batch['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long)
        
        image = batch['pixel_values'].to(device) # num_examples, max_images, images_color, image_X_dim, image_Y_dim
        num_images = batch['num_images'].to(device, dtype = torch.long)
        
        logits = model(
            input_ids = ids, 
            attention_mask = mask, 
            pixel_values = image, 
            num_images = num_images, 
            decoder_input_ids=y_ids, 
            labels=lm_labels
        )
        
        loss = logits[0]

        train_total_loss += loss.item()
        if idx%200 == 0:
            print({"Training Loss": loss.item()})
            
        (loss / accumulation_steps).backward()
        
        # update the weights only after accumulating k small batches (steps)
        if (idx + 1) % accumulation_steps == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(loader)
    
    # training time end
    training_time = time.time() - total_t0
    
    # print result summaries
    print("===============================================")
    print(" Training Results ")
    print("===============================================")
    print(f"EPOCH {epoch+1:1d} TRAIN done: - loss {avg_train_loss:.5f}")


def validate(epoch, tokenizer, model, device, loader):
    
    total_t0 = time.time()
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
            
            ids = batch['source_ids'].to(device, dtype = torch.long) # findings
            mask = batch['source_mask'].to(device, dtype = torch.long)

            y = batch['target_ids'].to(device, dtype = torch.long) # imp
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            
            image = batch['pixel_values'].to(device) # num_examples, max_images, images_color, image_X_dim, image_Y_dim
            num_images = batch['num_images'].to(device, dtype = torch.long)
            
            logits = model(
                input_ids = ids, 
                attention_mask = mask, 
                pixel_values = image, 
                num_images = num_images, 
                decoder_input_ids=y_ids, 
                labels=lm_labels
            )
            
            loss = logits[0]
            val_loss.append(loss)

            ###############################
            start_id = tokenizer.encode('<s>')[0]
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                pixel_values = image,
                num_images = num_images,
                max_length=128,
                num_beams=NUM_BEAMS,
                decoder_start_token_id=start_id,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                # do_sample=True, #  False, the model uses beam search to generate the next token in the sequence.
            )
            ###############################
            
            # Use the tokenizer to convert the output to a string
            # decoded preds	and labels in batch 
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            decoded_preds, decoded_target = postprocess_text(preds, target)
            rouge_score.add_batch(predictions=decoded_preds, references=decoded_target)

            predictions.extend(preds)
            actuals.extend(target)

            if step%200 == 0:
                print(f'Completed {step} step...')

        # Compute metrics
        avg_val_loss = statistics.fmean(val_loss)
        print("validation loss:", avg_val_loss)
        
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
        
    return predictions, actuals, rougel_f1