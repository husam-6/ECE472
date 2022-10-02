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

        self.cf10_train_labels = one_hot(np.array(cf10_labels[:40000]), 10)
        self.cf10_val_labels = one_hot(cf10_labels[40000:], 10)

        self.cf10_test = rgb_stack(cf10_test["data"]) / 255
        self.cf10_test_labels = one_hot(np.array(cf10_test["labels"]), 10)

        # CIFAR100 data
        cf100_data = cf100["data"]
        self.cf100_train = rgb_stack(cf100_data)[:40000, :, :, :] / 255
        self.cf100_val = rgb_stack(cf100_data)[40000:, :, :, :] / 255
        
        self.cf100_train_labels = cf100['fine_labels'][:40000]
        self.cf100_val_labels = cf100['fine_labels'][40000:]
        
        self.cf100_test = rgb_stack(cf100_test["data"]) / 255
        self.cf100_test_labels = np.array(cf100_test["fine_labels"])
    
    
    def augment_training_data(self, cifar=10, batch_size=32):
        """Function to augment (shuffle and rotate) the training data

        Obtained from https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
        """

        x_train = self.cf10_train
        y_train = self.cf10_train_labels
        if cifar == 100:
            x_train = self.cf100_train
            y_train = self.cf100_train_labels

        data = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

        training = data.flow(x_train, y_train, batch_size)
        steps_per_epoch = x_train.shape[0] // batch_size
        
        return training, steps_per_epoch


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


def one_hot(labels, k):
    """Convert label to one hot vector encodings to use label smoothing
    """
    output = np.zeros((labels.shape[0], k))
    for i, label in enumerate(labels):
        tmp = np.zeros(k)
        tmp[label] = 1
        output[i, :] = tmp

    return output


def unpickle(file):
    """Function to read in CIFAR data
    
    Obtained from https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    return {y.decode('ascii'): dict.get(y) for y in dict.keys()}


def relu_bn(inputs):
    """Helper function for model
    """
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def res_block(x, units, downsample=False):
    conv1 = layers.Conv2D(filters=units, kernel_size=(3,3), strides=(1 if not downsample else 2),
                            padding="same")(x)
    conv1 = relu_bn(conv1)
    conv2 = layers.Conv2D(filters=units, kernel_size=(3,3), strides=1, padding="same")(conv1)
    
    if downsample:
        x = layers.Conv2D(filters=units, kernel_size=1, strides=2, padding="same")(x)

    output = layers.Add()([x, conv2])
    output = relu_bn(output)
    
    return output


def create_model(classes, topk):
    """Creates keras model of a CNN

    Model loosely based off of example in 

    https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
    """

    inp = layers.Input(shape=(H, W, C))
    
    units = 64
    norm = layers.BatchNormalization()(inp)
    norm = layers.Conv2D(units, (3,3), padding="same")(norm) 
    
    # Start of first block
    block = relu_bn(norm)

    # Architecture of ResNet (2 blocks, then 5 blocks, then 2 blocks with the same num of units)
    blocks = [2, 5, 2]
    for i in range(len(blocks)):
        num_blocks = blocks[i]
        for j in range(num_blocks):
            block = res_block(block, units, downsample=(j == 0 and i != 0)) #Downsample every start of the next 'unit' of blocks
        units *= 2

    block = layers.AveragePooling2D((2,2), padding="same")(block)
    tmp = layers.Flatten()(block)

    # Add drop out
    tmp = layers.Dropout(0.2)(tmp)
    
    # L2 Regularization for last layer
    output = layers.Dense(classes, activation='softmax', activity_regularizer=tf.keras.regularizers.L2(0.001))(tmp)
    model = models.Model(inp, output)

    if topk != 1:
        # Default top k accuracy is k = 5
        metrics = ['accuracy', 'top_k_categorical_accuracy']
    else:
        metrics = ['accuracy']
    
    model.compile(optimizer="adam",
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                metrics=metrics
    )
    model.summary(print_fn=logging.info)
    return model


def main():
    """Main function for script execution"""
    
    # Set up logger and arguments
    args = parser.parse_args()

    EPOCHS = int(args.epochs)
    BATCHES = int(args.batch_size)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        filename='./hw4/output.txt',
        filemode='w'
	)
	
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console) 
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    

    # Process data
    logging.info("Reading in pickled data...")
    data = Data()

    # Apply CNN Model
    model = create_model(10, 1)
    fitted = model.fit(
        data.cf10_train, data.cf10_train_labels, epochs=EPOCHS,
        validation_data=(data.cf10_val, data.cf10_val_labels),
        batch_size=BATCHES
    )

    cf10_test_results = model.evaluate(data.cf10_test, data.cf10_test_labels, verbose=2)
    logging.info("On initial run of training set...")
    logging.info(f"Test set loss: {cf10_test_results[0]}")
    logging.info(f"Test set accuracy: {cf10_test_results[1]}")
    
    logging.info("Augmenting data and running training again")
    train_generator, steps = data.augment_training_data(10, BATCHES)
    
    fitted_second = model.fit(train_generator, validation_data=(data.cf10_val, data.cf10_val_labels),
            steps_per_epoch=steps, epochs=EPOCHS
    )

    # Test set
    cf10_test_results = model.evaluate(data.cf10_test, data.cf10_test_labels, verbose=2)
    logging.info(f"Test set loss: {cf10_test_results[0]}")
    logging.info(f"Test set accuracy: {cf10_test_results[1]}")

    # Plot image as example
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    
    #logging.info(data.cf10_train_labels[0])
    #logging.info(np.flatnonzero(data.cf10_train_labels[0]))
    
    label = data.cf10_label_names[np.flatnonzero(data.cf10_train_labels[0])[0]]
    label2 = data.cf100_label_names[np.flatnonzero(data.cf100_train_labels[0])[0]]
    
    ax[0].imshow(data.cf10_train[0, :, :, :])
    ax[0].set_title(f"CIFAR10: {label}")

    ax[1].imshow(data.cf100_train[0, :, :,:])
    ax[1].set_title(f"CIFAR100: {label2}")

    plt.savefig(f"{script_path}/cifar.pdf")

    loss = np.concatenate([fitted.history["loss"], fitted_second.history["loss"]])
    val_loss = np.concatenate([fitted.history["val_loss"], fitted_second.history["val_loss"]])

    plt.figure()
    plt.plot(loss, label = "Training")
    plt.plot(val_loss, label = "Validation")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(f"{script_path}/cifar10_loss.pdf")


if __name__ == "__main__":
    main()
