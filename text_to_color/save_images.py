"""
Final Project - Husam Almanakly and Michael Bentivegna

"""

# %% Libraries
import os
import requests
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
# import tensorflow as tf
from PIL import Image, ImageOps 
import PIL
from tqdm import tqdm
import grayscale
from io import BytesIO


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


size = (256, 256, 3)
def load_image(url):
    # Connect to url (if possible)
    try:
        res = requests.get(url)
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection refused at url {url}")
        return None
    if res.status_code != 200:
        logging.info(f"Could not connect to url {url}")
        return None
    
    # Read bytes from url (if possible)
    try:
        img = Image.open(BytesIO(res.content))
    except PIL.UnidentifiedImageError:
        logging.info(f"Could not read bytes from url {url}")
        return None
    
    # Resize / pad to be uniform (256, 256, 3)
    if np.array(img).shape != size:
        img_arr = np.array(resize_with_padding(img, size))
    else: 
        img_arr = np.array(img)
    
    if len(img_arr.shape) > 2 and img_arr.shape[2] == 4:
        #slice off the alpha channel
        img_arr = img_arr[:, :, :3]
    return img_arr


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    df = pd.read_parquet("part1.parquet")
    
    print(df.shape)
    images = []
    grayscale_images = []
    captions = []
    
    # Loop through dataset, download images, save grayscale and captions
    for i, row in df.iterrows():
        if i == 300000:
            break
        # Download image
        img = load_image(row["URL"])
        if img is None:
            continue

        # Get grayscale image 
        if len(img.shape) > 2:
            gray = grayscale.rgb2gray(img)
        else:
            gray = img
        if i % 100 == 0:
            logging.info(f"Downloading image: {i}")
        
        images.append(img)
        grayscale_images.append(gray)
        captions.append(row["TEXT"])
    # print(df["URL"].iloc[0])

    with open('images.npy', 'wb') as f:
        images = np.stack(images)
        np.save(f, images)
        logging.info(f"Images shape: {images.shape}")
    
    with open('captions.npy', 'wb') as f:
        captions = np.array(captions)
        np.save(f, captions)
        logging.info(f"Captions shape: {len(captions)}")
    
    with open('gray.npy', 'wb') as f:
        grayscale_images = np.stack(grayscale_images)
        np.save(f, grayscale_images)
        logging.info(f"Grayscale shape: {grayscale_images.shape}")


if __name__ == "__main__":
    main()
