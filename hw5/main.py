"""
Homework assignment 5 - Husam Almanakly

"""

# %% Libraries
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass, field, InitVar
from typing import Tuple
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig
from tqdm import tqdm
from transformers import TFDistilBertModel

MODEL_NAME = 'distilbert-base-uncased'
script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=31415, help="Random seed")
parser.add_argument("--epochs", default=1, help="Number of Epochs")
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
        
        cutoff = int(self.train_init["data"].to_numpy().shape[0] * 0.8)  # Cut off at 80%

        # Training Data
        self.train = self.train_init["data"].to_numpy()[:cutoff]
        self.train_labels = self.train_init["labels"].to_numpy()[:cutoff]
        
        self.val = self.train_init["data"].to_numpy()[cutoff:]
        self.val_labels = self.train_init["labels"].to_numpy()[cutoff:]
        
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


def tokenize(sentences, tokenizer, maxlen):
    """Tokenize input data
    
    https://www.kaggle.com/code/atechnohazard/news-classification-using-huggingface-distilbert
    """
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')


def fine_tune(data, batch_size, epochs, maxlen):
    """Using distilibert classifier. Code used from:

    https://medium.com/geekculture/hugging-face-distilbert-tensorflow-for-custom-text-classification-1ad4a49e26a7

    https://www.kaggle.com/code/atechnohazard/news-classification-using-huggingface-distilbert
    """

    # Tokenizer - given sentence, output corresponding token
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True, 
                    add_special_tokens=True, max_length=maxlen, pad_to_max_length=True)
    
    # Tokenize input data
    train_encodings = tokenize(data.train, tokenizer, maxlen)
    val_encodings = tokenize(data.val, tokenizer, maxlen)
    test_encodings = tokenize(data.test, tokenizer, maxlen)

    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(MODEL_NAME, config=config)

    input_ids_in = tf.keras.layers.Input(shape=(maxlen,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(maxlen,), name='masked_token', dtype='int32') 

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(4, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)

    for layer in model.layers[:3]:
        layer.trainable = False

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train - 'Fine Tuning' of pre-trained Distilibert dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings, data.train_labels))
    train_dataset = train_dataset.shuffle(200).batch(32)
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=(val_encodings, data.val_labels))

    # Evaluate on test set
    evaluation = model.evaluate(test_encodings, data.test_labels, batch_size=batch_size, verbose=2, return_dict=True)
    
    return evaluation


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
        filename='./hw5/output.txt',
        filemode='w'
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
    MAXLEN = (train["data"]).map(lambda x: len(x.split())).max()

    evaluation = fine_tune(data, BATCHES, EPOCHS, MAXLEN)

    logging.info(evaluation)


if __name__ == "__main__":
    main()