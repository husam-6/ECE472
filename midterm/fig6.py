"""
Husam Almanakly & Michael Bentivegna

Midterm Project 
Understanding intermediate layers using linear classifier probes
Figure 6
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
import probes

memory = Memory(".cache")

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=5, help="Number of Epochs")
parser.add_argument("--debug", default=False, help="Set logging level to debug")

def dense_model():
    """
    Function to create the model used to identify bad training using linear probes 
    128 Dense Layers with 128 units each
    """
    inp = layers.Input(shape=(28, 28, 1))
    flat = layers.Flatten()(inp)
    dense_init = layers.Dense(128)(flat)
    dense = layers.Dense(128)(dense_init)

    # 128 Dense Layers (intentional bad model for MNIST)
    for i in range(126):
        if i == 62:
            dense = layers.Add()([dense_init, dense])
        dense = layers.Dense(128)(dense)
            
    model = models.Model(inp, dense)
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
    )

    return model


def plot_bar(idx, i, string, ax, accuracy):
    """Helper function to plot the bar graphs of figure 6"""
    ax[i].bar(idx, 1 - accuracy, color="blue", width=1.0)
    ax[i].set_title(string)
    ax[i].set_xlabel("Layer")
    ax[i].set_ylabel("Error")
    ax[i].set_ylim([0, 1])
    ax[i].set_xlim([0, 140])



def main():
    """Main for script execution"""
    # Set up logger and arguments
    args = parser.parse_args()

    EPOCHS = int(args.epochs)

    # Get data (same MNIST dataset as before)
    data = probes.Data()

    # Initialize the model
    model = dense_model()
    
    # Probe each layer initially after small amount of training (2 Epochs)
    fitted = model.fit(
        data.train, data.train_labels, epochs=1,
        validation_data=(data.validation, data.validation_labels)
    )

    epoch1_accuracy = probes.check_probes(data, model, EPOCHS, False)

    # Probe after longer training (10 Epochs)
    fitted = model.fit(
        data.train, data.train_labels, epochs=3,
        validation_data=(data.validation, data.validation_labels)
    )
    
    # Probe each layer after training
    epoch4_accuracy = probes.check_probes(data, model, EPOCHS, False)
    
    fig, ax = plt.subplots(1, 2, figsize=(17, 10))
    idx = np.arange(0, len(epoch1_accuracy))

    # Plot results
    plot_bar(idx, 0, "Probes After 1 Epoch", ax, epoch1_accuracy)
    plot_bar(idx, 1, "Probes After 3 Epochs", ax, epoch4_accuracy)
          
    plt.savefig(f"{script_path}/figure6.pdf")

if __name__ == "__main__":
    main()
