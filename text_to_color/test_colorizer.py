""" Husam Almanakly and Michael Bentivegna

File to load in trained weights and test model with custom image + caption pair
"""

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from glob import glob
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import InstanceNormalization
from tqdm import tqdm
import gdown
from zipfile import ZipFile
import tfrecord
from os.path import exists
import logging
from skimage import io, color
from skimage.transform import resize
from transformers import T5Model, T5Tokenizer 
import train_colorizer


def colorize(path_to_file, path_to_img, t5_enc, t5_tok):
    file_exists = exists(path_to_file)
    rgb_exists = exists(path_to_img)
    if not file_exists or not rgb_exists:
        logging.info(f"Image doesn't exist")
        return
    with open(path_to_file) as f:
        lines = f.readlines()
    
    # Get Images
    rgb = resize(io.imread(path_to_img), (256, 256))
    gray = color.rgb2gray(rgb)
    assert gray.shape == (256, 256)

    # Pass tokens forward through encoder network
    enc = t5_tok(lines, max_length=1024, truncation=True, return_tensors="pt")
    output = t5_enc.encoder(
        input_ids=enc["input_ids"], 
        attention_mask=enc["attention_mask"], 
        return_dict=True
    )

    # Get the final hidden states
    emb = output.last_hidden_state.detach().numpy().squeeze()
    fixed_emb = emb.mean(axis=0)

    # Load in pre-trained colorizer model
    ckpt = tf.train.Checkpoint()

    # Define the models to restore
    input_shape = (256, 256, 1)
    model = train_colorizer.model_block(32, input_shape=input_shape, is_base=True, output=True)
    map_layer = train_colorizer.Mapping(1)

    # Restore the models from the checkpoint
    ckpt.restore("./checkpoints/ckpt-5.data-00000-of-00001").mapped = {"model": model, "mapping": map_layer}

    # Forward Prop to get final image
    embedding_w = map_layer(tf.squeeze(tf.convert_to_tensor(fixed_emb)))
    pred_ab = model([tf.reshape(tf.convert_to_tensor(gray), (-1, 256, 256, 1)), tf.squeeze(embedding_w)])

    # Add L channel from grayscale image to output
    pred_ab = pred_ab.numpy()
    lab_pred = np.dstack(np.resize(gray, (256, 256, 1)), pred_ab)

    return color.lab2rgb(lab_pred)


def main():
    logging.basicConfig(level=logging.INFO)

    # Remember to change to t5-3b when running on Kahan (those were the embeddings used in training)
    model = T5Model.from_pretrained("t5-large")
    tok = T5Tokenizer.from_pretrained("t5-large")
    
    path_to_file = f"./test/caption1.txt"
    path_to_img = f"./test/img1.jpeg"

    img = colorize(path_to_file, path_to_img, model, tok)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()
    



if __name__ == "__main__":
    main()