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

TRAIN_SAMPLES = 50000

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", default=100, help="Number of samples in dataset")
parser.add_argument("--batch_size", default=16, help="Number of samples in batch")
parser.add_argument("--num_iters", default=300, help="Number of SGD iterations")
parser.add_argument("--learning_rate", default=0.1, help="Learning rate / step size for SGD")
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--sigma_noise", default=0.5, help="Standard deviation of noise random variable")
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


    def get_training_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        index = np.arange(self.train.shape[0])
        choices = rng.choice(index, size=batch_size)

        return self.train[choices], self.train_labels[choices]


    def get_validation_batch(self, rng, batch_size):
        """
        Select random subset of examples for validation batch
        """
        index = np.arange(self.validation.shape[0])
        choices = rng.choice(index, size=batch_size)

        return self.validation[choices], self.validation_labels[choices]


# https://www.tensorflow.org/tutorials/images/cnn
def create_model():
    """Creates keras model of a CNN

    Code obtained from TensorFlow Docs example linked above
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
    model.add(layers.Dense(10, activity_regularizer=tf.keras.regularizers.L2(0.01)))

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
        level=logging.INFO
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate data
    seed_sequence = np.random.SeedSequence(args.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    num_samples = int(args.num_samples)
    batch_size = int(args.batch_size)

    # Process data - read in as csv using pandas, convert to numpy array
    # with proper dimensions
    df = pd.read_csv(f"{script_path}/mnist_train.csv")
    test = pd.read_csv(f"{script_path}/mnist_test.csv")
    validation = df.iloc[50000:]
    train = df.iloc[:50000]
    
    data = Data(train, validation, test)
    
    # logging.info(data.train.shape)
    # logging.info(labels)
    # sample = data.train[-1, :, :, :]
    # pixels = sample.reshape((28, 28))
    # plt.imshow(pixels, cmap="gray")
    
    # Fit model for a CNN 
    model = create_model()
    model.summary()
    fitted = model.fit(data.train, data.train_labels, epochs=10,
                       validation_data=(data.validation, data.validation_labels)
    )
    
    # Test model
    plt.figure()
    plt.plot(fitted.history["loss"], label="Training Loss")
    plt.plot(fitted.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Test on test set 
    test_results = model.evaluate(data.test, data.test_labels, verbose=2)
    logging.info(f"Test set loss: {test_results[0]}")
    logging.info(f"Test set accuracy: {test_results[1]}")

    plt.savefig(f"{script_path}/mnist_classify.pdf")


if __name__ == "__main__":
    main()
