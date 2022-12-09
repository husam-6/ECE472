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
import tensorflow as tf
import tfrecord


def main():
    # Get T5 Model
    model = T5Model.from_pretrained("t5-3b")
    tok = T5Tokenizer.from_pretrained("t5-3b")
    # Set logger
    logging.basicConfig(level=logging.INFO)
    get_embeddings(model, tok)


def get_embeddings(model, tok):
    record = []
    for i in range(3):
        # Get paths to image and caption
        path_to_file = f"./laion/captions/caption{i}.txt"
        path_to_img = f"./laion/images/img{i}.jpeg"
        file_exists = exists(path_to_file)
        rgb_exists = exists(path_to_img)
        if not file_exists or not rgb_exists:
            print("Image doesnt exist")
            continue
        with open(path_to_file) as f:
            lines = f.readlines()
        
        # Get Images
        rgb = resize(io.imread(path_to_img), (256, 256))
        gray = color.rgb2gray(rgb)
        labrgb = color.rgb2lab(rgb)
        assert labrgb.shape == (256, 256, 3)
        assert gray.shape == (256, 256)

        # Pass tokens forward through encoder network
        enc = tok(lines, max_length=1024, truncation=True, return_tensors="pt")
        output = model.encoder(
            input_ids=enc["input_ids"], 
            attention_mask=enc["attention_mask"], 
            return_dict=True
        )

        # Get the final hidden states
        emb = output.last_hidden_state.detach().numpy().squeeze()
        fixed_emb = emb.mean(axis=0)
        assert fixed_emb.shape[0] == 1024

        record.append(tfrecord.image_example(labrgb, gray, fixed_emb))
        logging.info(f"Creating TFRecord for object: {i}") 
        
        # Write to tfrecords file in batches
        if i == 2:
            # Write to tfrecords file 
            record_file = 'images.tfrecords'
            with tf.io.TFRecordWriter(record_file) as writer:
                for i in range(3):
                    tf_example = record[i]
                    writer.write(tf_example.SerializeToString())
            # Reset record
            record = []


if __name__ == "__main__":
    main()
