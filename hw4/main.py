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
    train_init: np.ndarray 
    test_init: np.ndarray  
    labels_init: np.ndarray 
    label_names: list 
    classes: int

    # Training data
    train: np.ndarray = field(init=False)
    train_labels: np.ndarray = field(init=False)
    
    # Validation Set
    val: np.ndarray = field(init=False)
    val_labels: np.ndarray = field(init=False)
    
    # Test data
    test: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)


    def __post_init__(self):
        """Data should be in N x H x W x C format 
        
        N = num samples
        H x W is the dimensions of the image
        C is the color channel (ie RGB => C = 3)
        """
        
        # CIFAR10 Training Data - use last 10,000 samples as validation set
        self.train = self.train_init[:40000, :, :, :] / 255
        self.val = self.train_init[40000:, :, :, :] / 255

        self.train_labels = one_hot(np.array(self.labels_init[:40000]), self.classes)
        self.val_labels = one_hot(np.array(self.labels_init[40000:]), self.classes)

        self.test = rgb_stack(self.test_init["data"]) / 255

        key = "labels"
        if self.classes == 100:
            key = "fine_labels"
        self.test_labels = one_hot(np.array(self.test_init[key]), self.classes)
    
    
    def augment_training_data(self, batch_size=32):
        """Function to augment (shuffle and rotate) the training data

        Obtained from https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
        """

        data = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

        training = data.flow(self.train, self.train_labels, batch_size)
        steps_per_epoch = self.train.shape[0] // batch_size
        
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
    bn = layers.BatchNormalization()(inputs)
    relu = layers.ReLU()(bn)
    return relu


def res_block(x, units, downsample=False):
    """Function with convolutional layers with residual connections

    Also obtained from https://www.cs.toronto.edu/~kriz/cifar.html
    """
    x = relu_bn(x) # Pre activation
    conv1 = layers.Conv2D(filters=units, kernel_size=(3,3), strides=(1 if not downsample else 2),
                            padding="same")(x)
    conv1 = relu_bn(conv1)
    
    conv2 = layers.Conv2D(filters=units, kernel_size=(3,3), strides=1, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)

    if downsample:
        x = layers.Conv2D(filters=units, kernel_size=1, strides=2, padding="same")(x)

    output = layers.Add()([x, conv2])
    output = layers.ReLU()(output)
    
    return output


def create_model(classes, topk):
    """Creates keras model of a CNN

    Model loosely based off of example in 

    https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
    """

    inp = layers.Input(shape=(H, W, C))
    
    units = 128
    norm = layers.BatchNormalization()(inp)
    norm = layers.Conv2D(units, (3,3), padding="same")(norm) 
    
    # Start of first block
    block = relu_bn(norm)

    # Architecture of network
    blocks = [2, 5, 2]
    for i in range(len(blocks)):
        num_blocks = blocks[i]
        for j in range(num_blocks):
            block = res_block(block, units, downsample=(j == 0 and i != 0)) #Downsample every start of the next 'unit' of blocks
        units *= 2

    block = layers.AveragePooling2D((2,2), padding="same")(block)
    tmp = layers.Flatten()(block)

    # Hidden layer
    tmp = layers.Dense(1024, activation = "relu")(tmp)
    
    # Add drop out
    tmp = layers.Dropout(0.1)(tmp)

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
    cf10, cf10_labels = unpack_cf10_data()
    cf100 = unpickle(f"{script_path}/cifar-100-python/train")
    cf100_data = rgb_stack(cf100["data"])
    cf10_test = unpickle(f"{script_path}/cifar-10-batches-py/test_batch")
    cf100_test = unpickle(f"{script_path}/cifar-100-python/test")
    
    # Label names
    cf10_label_names = unpickle(f"{script_path}/cifar-10-batches-py/batches.meta")["label_names"]
    cf100_label_names = unpickle(f"{script_path}/cifar-100-python/meta")["fine_label_names"]
    
    # Data objects for each dataset
    cifar10 = Data(cf10, cf10_test, cf10_labels, cf10_label_names, 10)
    cifar100 = Data(rgb_stack(cf100["data"]), cf100_test, cf100["fine_labels"], cf100_label_names, 100)

    
    def fit_model(model, data, label):
        
        # Obtained from 
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

        checkpoint_filepath = f'{script_path}/hw4/tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode='min',
            save_best_only=True
        )

        fitted = model.fit(
            data.train, data.train_labels, epochs=EPOCHS,
            validation_data=(data.val, data.val_labels), 
            batch_size=BATCHES, callbacks=[model_checkpoint_callback]
        )
        
        model.load_weights(checkpoint_filepath)

        test_results = model.evaluate(data.test, data.test_labels, verbose=2)
        val_acc = test_results[1]
        if label == 100:
            val_acc = test_results[2]
        
        logging.info("On initial run of training set...")
        logging.info(f"Test set loss: {test_results[0]}")
        
        statement = "Test set accuracy"
        if label == 100:
            statement = "Test set best of 5 accuracy: "
        
        logging.info(f"{statement}: {val_acc}")
        logging.info("Augmenting data and running training again")
        
        train_generator, steps = data.augment_training_data(BATCHES)
    
        fitted_second = model.fit(train_generator, validation_data=(data.val, data.val_labels),
                steps_per_epoch=steps, epochs=EPOCHS, callbacks=[model_checkpoint_callback]
        )
        
        model.load_weights(checkpoint_filepath)
        
        # Test set
        test_results = model.evaluate(data.test, data.test_labels, verbose=2) 
        val_acc = test_results[1]
        if label == 100:
            val_acc = test_results[2]

        logging.info(f"Test set loss: {test_results[0]}")
        logging.info(f"{statement}: {val_acc}")

        loss = np.concatenate([fitted.history["loss"], fitted_second.history["loss"]])
        val_loss = np.concatenate([fitted.history["val_loss"], fitted_second.history["val_loss"]])

        plt.figure()
        plt.plot(loss, label = "Training")
        plt.plot(val_loss, label = "Validation")
        plt.legend()
        plt.title("Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.savefig(f"{script_path}/cifar-{label}-loss.pdf")
    
    
    # Apply ResNet Model
    logging.info("Training model for CIFAR10 Dataset:")
    model = create_model(10, 1)
    fit_model(model, cifar10, 10) 
        
    logging.info("Training model for CIFAR100 Dataset: using top-5 accuracy")
    model = create_model(100, 5)
    fit_model(model, cifar100, 100)

    # Plot image as example
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    
    label = cifar10.label_names[np.flatnonzero(cifar10.train_labels[0, :])[0]]
    label2 = cifar100.label_names[np.flatnonzero(cifar100.train_labels[0, :])[0]]
    
    ax[0].imshow(cifar10.train[0, :, :, :])
    ax[0].set_title(f"CIFAR10: {label}")

    ax[1].imshow(cifar100.train[0, :, :,:])
    ax[1].set_title(f"CIFAR100: {label2}")

    plt.savefig(f"{script_path}/cifar.pdf")



if __name__ == "__main__":
    main()
