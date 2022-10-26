"""
Husam Almanakly & Michael Bentivegna

Midterm Project 
Understanding intermediate layers using linear classifier probes
Shared functions for both figures
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


@dataclass
class Data:
    """
    Data Class for MNIST Data obtained from Kaggle in CSV format
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


def check_probes(data, model, EPOCHS, fig3=True):
    """Helper function to train probes at each layer (other than flatten)"""
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


def probes(data, input_layer, epochs, first=False):
    """Trains a linear probe given the graph for the input layer and the data class"""
    if not first:
        train = input_layer(data.train)[0]
        val = input_layer(data.validation)
        test = input_layer(data.test)
    else:
        train = data.train
        val = data.validation
        test = data.test
    
    # 1 Dense layer linear classifier
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