""" Husam Almanakly and Michael Bentivegna

File to load in trained weights and test model with custom image + caption pair
"""

import os
import argparse
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

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="./laion/captions/caption0.txt", help="Caption File")
parser.add_argument("--image_path", default="./laion/images/img0.jpeg", help="Image Path")
parser.add_argument("--checkpoints", default="./checkpoints_normalized", help="Checkpoints Directory")


def colorize(path_to_file, path_to_img, checkpoint, t5_enc, t5_tok):
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
    model = train_colorizer.model_block(32, input_shape=input_shape, is_base=True, output=True, blocks=6)
    map_layer = train_colorizer.Mapping(1, 1024)

    # Restore the models from the checkpoint
    ckpt.mapped = {'model': model, "mapping": map_layer}
    ckpt.restore(tf.train.latest_checkpoint(checkpoint))

    # Forward Prop to get final image
    emb_tensor = tf.convert_to_tensor(fixed_emb)
    emb_tensor = tf.reshape(emb_tensor, (1, 1024))
    embedding_w = map_layer(emb_tensor)
    embedding_w = tf.reshape(embedding_w, (1, 512))
    inp = tf.reshape(tf.convert_to_tensor(gray, dtype=tf.float64), (-1, 256, 256, 1)) 
    pred_ab = model([inp, embedding_w])

    # Add L channel from grayscale image to output
    pred_ab = pred_ab.numpy().squeeze() * 128
    gray = color.gray2rgb(gray)
    gray = color.rgb2lab(gray)
    # print(gray.shape)
    # print(pred_ab.shape)
    lab_pred = np.dstack((gray[:, :, 0], pred_ab))

    return color.lab2rgb(lab_pred)


def main():
    # Set up Logger
    logging.basicConfig(level=logging.INFO)
    
    # Get arguments
    args = parser.parse_args()

    # Remember to change to t5-3b when running on Kahan (those were the embeddings used in training)
    model = T5Model.from_pretrained("t5-3b")
    tok = T5Tokenizer.from_pretrained("t5-3b")
    
    start_idx = 150_001
    end_idx = 150_101
    image_prefix = "./laion/images/img"
    file_prefix = "./laion/captions/caption"
    for i in range(start_idx, end_idx):
        isExist = os.path.exists(f"{image_prefix}{i}.jpeg")
        if not isExist:
            logging.info(f"File with index {i} doesn't exist")
            continue
        img = colorize(f"{file_prefix}{i}.txt", f"{image_prefix}{i}.jpeg", args.checkpoints, model, tok)
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.savefig(f"./output/test_{i}.jpeg", format="jpeg", bbox_inches='tight')
        logging.info(f"Saving colorized image to file ./output/test_{i}.pdf...")


if __name__ == "__main__":
    main()
