"""
Homework assignment 4 - Husam Almanakly

"""

import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
from tqdm import trange
from dataclasses import dataclass, field, InitVar
import pickle

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")


# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=31415, help="Number of Epochs")
parser.add_argument("--debug", default=False, help="Set logging level to debug")


def unpickle(file):
    """Function to read in CIFAR data
    
    Obtained from https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return {y.decode('ascii'): dict.get(y) for y in dict.keys()}


def create_model():
    """Creates keras model of a CNN

    Code obtained from TensorFlow Docs example linked below
    
    https://www.tensorflow.org/tutorials/images/cnn
    """

    model = models.Sequential()

    # CNN Layers
    model.add(layers.Conv2D(15, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(15, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(15, (3, 3), activation='relu'))
    
    # Add Dense Layer for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(15, activation='relu'))

    # Add drop out
    model.add(layers.Dropout(0.5))
    
    # L2 Regularization for last layer
    model.add(layers.Dense(10, activity_regularizer=tf.keras.regularizers.L2(0.001)))

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
    )
    return model



def main():
    """Main function for script execution"""
    
    # Set up logger and arguments
    args = parser.parse_args()

    EPOCHS = int(args.epochs)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process data
    labels = []
    images = []
    for batch in range(1, 6):
        logging.info(f"Unpacking batch {batch}")
        unpickled_data = unpickle(f"{script_path}/cifar-10-batches-py/data_batch_{batch}")
        # logging.info(data_test['data'])
        im = unpickled_data["data"]
        labels.append(unpickled_data["labels"])
        
        im_r = im[:, 0:1024].reshape(-1, 32, 32)
        im_g = im[:, 1024:2048].reshape(-1, 32, 32)
        im_b = im[:, 2048:].reshape(-1, 32, 32)

        images.append(np.stack([im_r, im_g, im_b], axis=-1))

    labels = np.concatenate(labels, axis=0)
    images = np.concatenate(images, axis=0)
    logging.info(f"Class: {labels[0]}")
    logging.info(images.shape)

    # Plot image as example
    plt.figure()
    plt.imshow(images[0, :, :, :])
    plt.savefig(f"{script_path}/cifar.pdf")


if __name__ == "__main__":
    main()
