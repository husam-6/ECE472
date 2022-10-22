"""
Midterm Project - Husam Almanakly & Michael Bentivegna

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
from keras import backend as K
from joblib import Memory

memory = Memory(".cache")

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")

tf.compat.v1.disable_eager_execution()

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=5, help="Number of Epochs")
parser.add_argument("--debug", default=False, help="Set logging level to debug")


@dataclass
class Data:
    """Data Class for MNIST Data obtained from Kaggle in CSV format
    
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv
    """
    
    # Training data
    train: np.ndarray = field(init=False)
    train_labels: np.ndarray = field(init=False)
    
    # Validation Set
    validation: np.ndarray = field(init=False)
    validation_labels: np.ndarray = field(init=False)

    # Test data
    test: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)


    def __post_init__(self):
        """Data should be in N x H x W x C format 
        
        N = num samples
        H x W is the dimensions of the image
        C is the color index which is 1 for now since MNIST is grayscale
        """

        # Process data - read in as csv using pandas, convert to numpy array
        # with proper dimensions
        logging.info("Processing data...")
        df = pd.read_csv(f"{script_path}/mnist_train.csv")
        test = pd.read_csv(f"{script_path}/mnist_test.csv")
        validation = df.iloc[55000:]
        train = df.iloc[:55000]

        self.train_labels = train.values[:, 0]
        self.train = train.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
        
        self.test_labels = test.values[:, 0]
        self.test = test.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
        
        self.validation_labels = validation.values[:, 0]
        self.validation = validation.drop("label", axis=1).values.reshape(-1, 28, 28, 1)


def create_model():
    """Creates keras model of a CNN

    Code obtained from TensorFlow Docs example linked below
    
    https://www.tensorflow.org/tutorials/images/cnn
    """

    model = models.Sequential()

    # CNN Layers
    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.ReLU())
    model.add(layers.Dense(10))

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
    )

    return model


def probes(data, input_layer, epochs, first=False):
    if not first:
        train = input_layer(data.train)[0]
        val = input_layer(data.validation)
        test = input_layer(data.test)
    else:
        train = data.train
        val = data.validation
        test = data.test

    # print(train[0].shape)
  
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(10))
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
    )

    hist = model.fit(train, data.train_labels, validation_data=(val, data.validation_labels), epochs=epochs, verbose=0)
    
    res = model.evaluate(test, data.test_labels)

    return res[1]


def plotting_probes(idx, labels, i, string, ax, accuracy):
    ax[i].plot(idx, 1 - accuracy, linewidth=1.8, color="deeppink")
    ax[i].set_title(string)
    ax[i].set_xlabel("Layer")
    ax[i].set_ylabel("Error")
    ax[i].set_xticks(idx, labels, rotation = 30)
    ax[i].set_xlim([0, 11])
    ax[i].set_ylim([0, 0.2])  

def check_probes(data, model, EPOCHS, fig3=True):
    # Check the first probe
    inp = model.input                          
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

    # Get rid of flatten layer
    if fig3:
        functors.pop(7) 
    else:
        functors.pop(0) 

    # Train probe on regular input data
    accuracy = np.zeros(len(functors) + 1)
    accuracy[0] = probes(data, None, EPOCHS, True)
    print(f"Training Probe 0")

    # Probe each layer as a classifier
    for i, layer in enumerate(functors):
        print(f"Training Probe {i + 1}")
        layer_acc = probes(data, layer, EPOCHS)
        accuracy[i+1] = layer_acc
    
    return accuracy

def main():
    """Main function for script execution"""
    
    # Set up logger and arguments
    args = parser.parse_args()

    EPOCHS = int(args.epochs)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        filename='hw3/output.txt'
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    
    data = Data()
    
    # Fit model for a CNN 
    model = create_model()
    model.summary(print_fn=logging.info)

    # Fit model based on training set
    model = create_model()
    
    # Probe each layer initially with no training
    prev_accuracy = check_probes(data, model, EPOCHS)

    fitted = model.fit(
        data.train, data.train_labels, epochs=EPOCHS,
        validation_data=(data.validation, data.validation_labels)
    )
    
    # Probe each layer after training
    final_accuracy = check_probes(data, model, EPOCHS)
    
    labels = ["Input", "conv1_preact", "conv1_postact", "conv1_postpool",
                      "conv2_preact", "conv2_postact", "conv2_postpool", "fc1_preact",
                      "fc1_postact", "logits"]
    
    fig, ax = plt.subplots(1, 2, figsize=(17, 10))
    idx = np.arange(1, 11)

    # Plot results
    plotting_probes(idx, labels, 0, "After Initialization, No Training", ax, prev_accuracy)
    plotting_probes(idx, labels, 1, "After Training for 10 Epochs", ax, final_accuracy)
          
    plt.savefig(f"{script_path}/figure3.pdf")


if __name__ == "__main__":
    main()
