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
    
    logging.info(data.train.shape)
    
    
    # vals = train.values.reshape(-1, 28, 28, 1)
    
    # logging.info(labels)
    # sample = vals[-1, :, :, :]
    # pixels = sample.reshape((28, 28))
    # plt.imshow(pixels, cmap="gray")
    
    

    plt.savefig(f"{script_path}/mnist_classify.pdf")


if __name__ == "__main__":
    main()
