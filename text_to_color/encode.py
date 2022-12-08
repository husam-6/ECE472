"""
Script to encode each caption from Laion dataset images 
"""

from transformers import T5Model, T5Tokenizer 
import numpy as np
from os.path import exists
import logging
import time
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt


def main():
    # Get T5 Model
    model = T5Model.from_pretrained("t5-3b")
    tok = T5Tokenizer.from_pretrained("t5-3b")
    # Set logger
    logging.basicConfig(level=logging.INFO)
    get_embeddings(model, tok)

def get_embeddings(model, tok):
    # start = time.time()
    for i in range(1):
        path_to_file = f"./captions/caption{i}.txt"
        path_to_img = f"./laion/images/img{i}.jpeg"
        file_exists = exists(path_to_file)
        rgb_exists = exists(path_to_img)
        if not file_exists or not gray_exists or not rgb_exists:
            continue
        with open(path_to_file) as f:
            lines = f.readlines()
        
        # Get Images
        rgb = resize(io.imread(path_to_img), (256, 256))
        gray = color.rgb2gray(rgb)
        labrgb = color.rgb2lab(rgb)
        # print(labrgb.shape)
        # print(gray.shape)
        
        # Pass tokens forward through encoder network
        enc = tok(lines, max_length=1024, truncation=True, return_tensors="pt")
        
        plt.figure()
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(gray, cmap="gray")
        ax[1].imshow(rgb, cmap="gray")
        plt.savefig("./data_test.pdf")
        # forward pass through encoder only
        output = model.encoder(
            input_ids=enc["input_ids"], 
            attention_mask=enc["attention_mask"], 
            return_dict=True
        )

        # get the final hidden states
        emb = output.last_hidden_state.detach().numpy().squeeze()
        fixed_emb = emb.mean(axis=0)
        logging.info(f"Creating TFRecord for object: {i}") 
        assert fixed_emb.shape[0] == 1024
        embeddings.append(fixed_emb)
    # end = time.time()
    # print(end - start)
    



if __name__ == "__main__":
    main()
