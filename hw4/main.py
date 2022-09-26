"""
Homework assignment 4 - Husam Almanakly

"""

from ctypes.wintypes import HACCEL
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
import argparse
from tqdm import trange
from dataclasses import dataclass, field, InitVar
from typing import Tuple
import pickle

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")


# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=5, help="Number of Epochs")
parser.add_argument("--batch_size", default=32, help="Batch size for SGD")
parser.add_argument("--groups", default=32, help="Number of Groups in GroupNorm")
parser.add_argument("--debug", default=False, help="Set logging level to debug")

# Image dimensions
H = 32
W = 32
C = 3


@dataclass
class Data:
    """Data Class for CIFAR Data. Obtained from: 
    
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
   
    # Training data
    cf10_train: np.ndarray = field(init=False)
    cf10_train_labels: np.ndarray = field(init=False)
    
    cf100_train: np.ndarray = field(init=False)
    cf100_train_labels: np.ndarray = field(init=False)
    
    # Validation Set
    cf10_val: np.ndarray = field(init=False)
    cf10_val_labels: np.ndarray = field(init=False)
    
    cf100_val: np.ndarray = field(init=False)
    cf100_val_labels: np.ndarray = field(init=False)

    # Test data
    cf10_test: np.ndarray = field(init=False)
    cf10_test_labels: np.ndarray = field(init=False)
    
    cf100_test: np.ndarray = field(init=False)
    cf100_test_labels: np.ndarray = field(init=False)

    cf10_label_names: list = field(init=False)
    cf100_label_names: list = field(init=False)


    def __post_init__(self):
        """Data should be in N x H x W x C format 
        
        N = num samples
        H x W is the dimensions of the image
        C is the color channel (ie RGB => C = 3)
        """

        # Read in pickled data
        cf10, cf10_labels = unpack_cf10_data()
        cf100 = unpickle(f"{script_path}/cifar-100-python/train")
        cf10_test = unpickle(f"{script_path}/cifar-10-batches-py/test_batch")
        cf100_test = unpickle(f"{script_path}/cifar-100-python/test")
        
        # Label names
        cf10_label_names = unpickle(f"{script_path}/cifar-10-batches-py/batches.meta")["label_names"]
        cf100_label_names = unpickle(f"{script_path}/cifar-100-python/meta")["fine_label_names"]
        
        self.cf10_label_names = [y.decode("ascii") for y in cf10_label_names]
        self.cf100_label_names = [y.decode("ascii") for y in cf100_label_names]
        
        # CIFAR10 Training Data - use last 10,000 samples as validation set
        self.cf10_train = cf10[:40000, :, :, :] / 255
        self.cf10_val = cf10[40000:, :, :, :] / 255

        self.cf10_train_labels = cf10_labels[:40000]
        self.cf10_val_labels = cf10_labels[40000:]

        self.cf10_test = rgb_stack(cf10_test["data"]) / 255
        self.cf10_test_labels = np.array(cf10_test["labels"])

        # CIFAR100 data
        cf100_data = cf100["data"]
        self.cf100_train = rgb_stack(cf100_data)[:40000, :, :, :] / 255
        self.cf100_val = rgb_stack(cf100_data)[40000:, :, :, :] / 255
        
        self.cf100_train_labels = cf100['fine_labels'][:40000]
        self.cf100_val_labels = cf100['fine_labels'][40000:]
        
        self.cf100_test = rgb_stack(cf100_test["data"]) / 255
        self.cf100_test_labels = np.array(cf100_test["fine_labels"])


def unpack_cf10_data() -> Tuple[np.ndarray]:
    """Combine all 5 'batches' in CIFAR10 data"""
    labels = []
    cf10 = []
    for batch in range(1, 6):
        unpickled_data = unpickle(f"{script_path}/cifar-10-batches-py/data_batch_{batch}")
        im = unpickled_data["data"]
        labels.append(unpickled_data["labels"])

        cf10.append(rgb_stack(im))

    
    labels = np.concatenate(labels, axis=0)
    cf10 = np.concatenate(cf10, axis=0)

    return cf10, labels


def rgb_stack(im: np.ndarray) -> np.ndarray:
    """Function to reformat input Numpy array to be H x W x C
    
    Input should be an N x 3072 numpy array where N = number of 
    samples
    """
    assert im.shape[1] == 3072
    assert len(im.shape) == 2

    im_r = im[:, 0:1024].reshape(-1, 32, 32)
    im_g = im[:, 1024:2048].reshape(-1, 32, 32)
    im_b = im[:, 2048:].reshape(-1, 32, 32)

    return np.stack([im_r, im_g, im_b], axis=-1)


def unpickle(file):
    """Function to read in CIFAR data
    
    Obtained from https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    return {y.decode('ascii'): dict.get(y) for y in dict.keys()}


def create_model(classes):
    """Creates keras model of a CNN

    Code obtained from TensorFlow Docs example linked below
    
    https://www.tensorflow.org/tutorials/images/cnn
    """

    model = models.Sequential()

    # CNN Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(H, W, C)))
    model.add(layers.BatchNormalization(epsilon=1e-06, momentum=0.9))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization(epsilon=1e-06, momentum=0.9))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization(epsilon=1e-06, momentum=0.9))
    
    # Add Dense Layer for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))

    # Add drop out
    model.add(layers.Dropout(0.5))
    
    # L2 Regularization for last layer
    model.add(layers.Dense(classes, activity_regularizer=tf.keras.regularizers.L2(0.001)))

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
    GROUPS = int(args.groups)
    BATCHES = int(args.batch_size)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    

    # Process data
    logging.info("Reading in pickled data...")
    data = Data()

    # Apply CNN Model
    model = create_model(10)
    fitted = model.fit(
        data.cf10_train, data.cf10_train_labels, epochs=EPOCHS,
        validation_data=(data.cf10_val, data.cf10_val_labels),
        batch_size=BATCHES
    )

    # Test set
    cf10_test_results = model.evaluate(data.cf10_test, data.cf10_test_labels, verbose=2)
    logging.info(f"Test set loss: {cf10_test_results[0]}")
    logging.info(f"Test set accuracy: {cf10_test_results[1]}")

    # Plot image as example
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    label = data.cf10_label_names[data.cf10_train_labels[0]]
    label2 = data.cf100_label_names[data.cf100_train_labels[0]]
    
    ax[0].imshow(data.cf10_train[0, :, :, :])
    ax[0].set_title(f"CIFAR10: {label}")

    ax[1].imshow(data.cf100_train[0, :, :,:])
    ax[1].set_title(f"CIFAR100: {label2}")

    plt.savefig(f"{script_path}/cifar.pdf")


if __name__ == "__main__":
    main()
