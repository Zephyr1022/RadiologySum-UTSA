import os
import time
import torch
import numpy as np
import pandas as pd

# embeddings_train
def embeddings(epoch, tokenizer, model, device, loader, SAVE_PATH, split):
    
    print("")
    print("Extract Embeddings...")
    total_t0 = time.time()
    model.eval()

    text_embeddings = []
    image_embeddings = []
    multimodal_embeddings = []
    
    with torch.no_grad():
        for step, batch in enumerate(loader, 0):
            if step % 200 == 0:
                print("Extract Embeddings Process", step)
                sys.stdout.flush()
            
            ids = batch['source_ids'].to(device, dtype = torch.long) # findings
            mask = batch['source_mask'].to(device, dtype = torch.long)
            image = batch['pixel_values'].to(device) # num_examples, max_images, images_color, image_X_dim, image_Y_dim
            num_images = batch['num_images'].to(device)
            
            # Get the text embeddings
            text_embeds = model.encoder.get_text_encoder_embeds(ids, mask)
            text_embeds = torch.split(text_embeds, 1, dim=0)
            # text_embeddings.append(text_embed.detach().cpu().numpy().flatten())
            for text_embed in text_embeds:
                text_embeddings.append(text_embed.squeeze().cpu().detach().numpy())

            # Get the image embeddings
            image_embeds, _ = model.encoder.get_embeddings(image, num_images)
            image_embeds = torch.split(image_embeds, 1, dim=0)
            # image_embeddings.append(image_embed.detach().cpu().numpy().flatten().astype('float16'))
            for image_embed in image_embeds:
                image_embeddings.append(image_embed.squeeze().cpu().detach().numpy())
            
            # Get the multimodal embeddings
            multimodal_embeds = model.encoder.get_text_image_encoder_embeds(ids, mask, image, num_images)
            multimodal_embeds = torch.split(multimodal_embeds, 1, dim=0)
            for multimodal_embed in multimodal_embeds:
                multimodal_embeddings.append(multimodal_embed.squeeze().cpu().detach().numpy())
            
    text_embeddings = np.array(text_embeddings)
    image_embeddings = np.array(image_embeddings)
    multimodal_embeddings = np.array(multimodal_embeddings)
    
    print("Time taken to extract embeddings: {:.2f}s".format(time.time() - total_t0))
    print("Completed", split)
    print("Shape", text_embeddings.shape, image_embeddings.shape, multimodal_embeddings.shape)
    
    output_file_text =  SAVE_PATH +'text_'+'{}.npy'.format(split)
    output_file_image =  SAVE_PATH +'image_' +'{}.npy'.format(split)
    output_file_multimodal =  SAVE_PATH +'multimodal_' +'{}.npy'.format(split)
    
    print("output_file:",output_file_text,output_file_image,output_file_multimodal)
    np.save(output_file_text, text_embeddings)
    np.save(output_file_image, image_embeddings)
    np.save(output_file_multimodal, multimodal_embeddings)
    
    return None

    # The issue is keeping multiple copies of the data uses more space
    # If you dave in the same variable name some space can be saved
    
    