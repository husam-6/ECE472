"""
Homework assignment 5 - Husam Almanakly

"""

# Libraries
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
from transformers import AutoModel, AutoTokenizer

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=5, help="Number of Epochs")
parser.add_argument("--batch_size", default=32, help="Batch size for SGD")
parser.add_argument("--groups", default=32, help="Number of Groups in GroupNorm")
parser.add_argument("--debug", default=False, help="Set logging level to debug")


@dataclass
class Data:
    """Data Class for AG News Data. Obtained from: 
    
    https://huggingface.co/datasets/ag_news/tree/main

    Training
    https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
    
    Testing
    -> https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
    """

    train_init: np.ndarray 
    test_init: np.ndarray  
    label_names: list

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
        """Data should be text with an associated label (0 - 3)
        
        Category names include World, Sports, Business, Sci/Tech
        """
        
        # Training Data
        self.train = self.train_init["data"].to_numpy()
        self.train_labels = self.train_init["labels"].to_numpy()
        
        # Test Set
        self.test = self.test_init["data"].to_numpy()
        self.test_labels = self.test_init["labels"].to_numpy()


def process_ag_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Function to process the read csv file for AG news dataset
    """
    
    assert isinstance(df, pd.DataFrame)
    df["data"] = df[1] + df[2]
    df = df.rename({0: "labels"}, axis=1).drop([1, 2], axis=1)
    df["labels"] = df["labels"] - 1

    assert isinstance(df, pd.DataFrame)
    return df


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
    )
    
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process data
    logging.info("Reading in AG News dataset...")
    
    # Read in data
    train = pd.read_csv(f"{script_path}/train.csv", header=None)
    test = pd.read_csv(f"{script_path}/test.csv", header=None)

    train = process_ag_csv(train)
    test = process_ag_csv(test)

    data = Data(train, test, ["World", "Sports", "Business", "Sci/Tech"])

    # Using Distilibert Pre Trained NLP Model
    # https://autonlp.ai/models/distilbert-base-uncased
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print(type(model))
    print(type(tokenizer))

if __name__ == "__main__":
    main()
