"""
Husam Almanakly & Michael Bentivegna

Midterm Project 
Understanding intermediate layers using linear classifier probes
Figure 3
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

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=5, help="Number of Epochs")
parser.add_argument("--debug", default=False, help="Set logging level to debug")


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


def plotting_probes(idx, labels, i, string, ax, accuracy):
    """Helper function for plotting probe accuracy before and after training"""
    ax[i].plot(idx, 1 - accuracy, linewidth=1.8, color="deeppink")
    ax[i].set_title(string)
    ax[i].set_xlabel("Layer")
    ax[i].set_ylabel("Error")
    ax[i].set_xticks(idx, labels, rotation = 30)
    ax[i].set_xlim([0, 11])
    ax[i].set_ylim([0, 0.2])  


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
    
    
    data = probes.Data()
    
    # Fit model for a CNN 
    model = create_model()
    model.summary(print_fn=logging.info)

    # Fit model based on training set
    model = create_model()
    
    # Probe each layer initially with no training
    prev_accuracy = probes.check_probes(data, model, EPOCHS)

    fitted = model.fit(
        data.train, data.train_labels, epochs=EPOCHS,
        validation_data=(data.validation, data.validation_labels)
    )
    
    # Probe each layer after training
    final_accuracy = probes.check_probes(data, model, EPOCHS)
    
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
