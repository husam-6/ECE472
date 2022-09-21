"""
Homework assignment 3 - Husam Almanakly

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

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")


# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--debug", default=False, help="Set logging level to debug")


@dataclass
class Data:
    """Data Class for MNIST Data obtained from Kaggle in CSV format
    
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv
    """
    
    training_initial: pd.DataFrame
    validation_initial: pd.DataFrame
    test_initial: pd.DataFrame
    
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

        self.train_labels = self.training_initial.values[:, 0]
        self.train = self.training_initial.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
        
        self.test_labels = self.test_initial.values[:, 0]
        self.test = self.test_initial.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
        
        self.validation_labels = self.test_initial.values[:, 0]
        self.validation = self.test_initial.drop("label", axis=1).values.reshape(-1, 28, 28, 1)


def create_model():
    """Creates keras model of a CNN

    Code obtained from TensorFlow Docs example linked below
    
    https://www.tensorflow.org/tutorials/images/cnn
    """

    model = models.Sequential()

    # CNN Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Add Dense Layer for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

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

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        filename='hw3/output.txt'
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process data - read in as csv using pandas, convert to numpy array
    # with proper dimensions
    logging.info("Processing data...")
    df = pd.read_csv(f"{script_path}/mnist_train.csv")
    test = pd.read_csv(f"{script_path}/mnist_test.csv")
    validation = df.iloc[50000:]
    train = df.iloc[:50000]
    
    data = Data(train, validation, test)
    
    # Fit model for a CNN 
    model = create_model()
    model.summary(print_fn=logging.info)

    # Fit model based on training set
    model = create_model()
    fitted = model.fit(
        data.train, data.train_labels, epochs=5,
        validation_data=(data.validation, data.validation_labels)
    )
    
    # Test model
    plt.figure()
    plt.plot(fitted.history["loss"], label="Training Loss")
    plt.plot(fitted.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Test on test set 
    test_results = model.evaluate(data.test, data.test_labels, verbose=2)
    logging.info(f"Test set loss: {test_results[0]}")
    logging.info(f"Test set accuracy: {test_results[1]}")

    plt.savefig(f"{script_path}/mnist_classify.pdf")


if __name__ == "__main__":
    main()
