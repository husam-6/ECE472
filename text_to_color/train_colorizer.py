""" Husam Almanakly and Michael Bentivegna

DL Final Project
"""

import os
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from glob import glob
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import InstanceNormalization
from tqdm import tqdm
import gdown
from zipfile import ZipFile
import tfrecord
import logging


# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=5, help="Number of epochs")
parser.add_argument("--learning_rate", default=0.001, help="Learning Rate")
parser.add_argument("--batch_size", default=32, help="Batch Size")


def pixel_norm(x, epsilon=1e-8):
    return x / tf.math.sqrt(tf.reduce_mean(x ** 2, axis=-1, keepdims=True) + epsilon)


def Mapping(num_stages, input_shape=512):
    z = layers.Input(shape=(input_shape))
    w = pixel_norm(z)
    for i in range(8):
        w = EqualizedDense(512, learning_rate_multiplier=0.01)(w)
        w = layers.LeakyReLU(0.2)(w)
    w = tf.tile(tf.expand_dims(w, 1), (1, num_stages, 1))
    return keras.Model(z, w, name="mapping")


class AdaIN(layers.Layer):
    def __init__(self, gain=1, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.gain = gain

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = EqualizedDense(self.x_channels, gain=1)
        self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb


class EqualizedConv(layers.Layer):
    def __init__(self, out_channels, kernel=3, gain=2, strides=1, **kwargs):
        super(EqualizedConv, self).__init__(**kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel != 1
        self.strides = strides

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel, self.kernel, self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.kernel * self.kernel * self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        if self.pad:
            x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        else:
            x = inputs

        output = (
            tf.nn.conv2d(x, self.scale * self.w, strides=self.strides, padding="VALID") + self.b
        )
        return output


class EqualizedDense(layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier=1, **kwargs):
        super(EqualizedDense, self).__init__(**kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        output = tf.add(tf.matmul(inputs, self.scale * self.w), self.b)
        return output * self.learning_rate_multiplier


def block(x, w_embedding, filter_num):
        x = layers.LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w_embedding])

        x = EqualizedConv(filter_num, 3)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w_embedding])
        return x


def model_block(filter_num, input_shape, is_base=True, output=True, blocks=5):
    """ Generator / model block
    
    This function takes in an input and creates a block of our StyleGAN generator block like 
    architecture
    """
    # Grayscale image
    input_tensor = layers.Input(shape=input_shape)
    w_embedding = layers.Input(shape=512)
    x = input_tensor

    # Don't need noise for our model
    # x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu')(x)
    x = EqualizedConv(3, 3)(x)

    # Conv1 block
    x = EqualizedConv(filter_num, 3, strides=2)(x)
    x = block(x, w_embedding, filter_num)
        
    # Conv2 
    x = EqualizedConv(filter_num * 4, 3, strides=2)(x)
    x = block(x, w_embedding, filter_num * 4)

    # Conv3 
    x = EqualizedConv(filter_num * 4, 3, strides=2)(x)
    x = block(x, w_embedding, filter_num * 4)
    
    # Conv4
    x = EqualizedConv(filter_num * 4, 3, strides=2)(x)
    x = block(x, w_embedding, filter_num * 8)

    # Conv5
    x = block(x, w_embedding, filter_num * 8)
    
    # Conv6
    x = block(x, w_embedding, filter_num * 8)
    
    # Conv7 block 
    x = layers.UpSampling2D((2,2))(x)
    x = EqualizedConv(filter_num * 4, 3)(x)
    x = block(x, w_embedding, filter_num * 4)

    # Conv8 block 
    x = layers.UpSampling2D((2,2))(x)
    x = EqualizedConv(filter_num * 4, 3)(x)
    x = block(x, w_embedding, filter_num * 4)

    # Conv9 block 
    x = layers.UpSampling2D((2,2))(x)
    x = EqualizedConv(filter_num * 2, 3)(x)
    x = block(x, w_embedding, filter_num * 2)

    # Conv10 block 
    x = layers.UpSampling2D((2,2))(x)
    x = EqualizedConv(filter_num, 3)(x)
    x = block(x, w_embedding, filter_num)
    
    if output:
        # Get only 2 dimensions (for ab channels)
        x = EqualizedConv(2, 3)(x)
    
    return keras.Model([input_tensor, w_embedding], x)


def train(dataset, validation, model, map_layer, epochs, learning_rate, batch_size):
    """ Function to train custom StyleGAN-like Architecture"""
    logging.info(f"Training on {epochs} epochs") 
    # Set up Checkpoint Variables
    checkpoint_dir = "./checkpoints_normalized_tests/"

    # optimizer = tf.optimizers.Adam()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    # Add the models to the checkpoint object
    checkpoint = tf.train.Checkpoint()
    checkpoint.mapped = {"model": model, "mapping": map_layer}
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

    # loss func
    loss_fn = keras.losses.MeanSquaredError()

    saved_loss=[]
    loss_epochs=[]
    saved_val_loss=[]
    # Train the model
    for i in range(epochs):
        logging.info(f"Epoch {i + 1} / {epochs}")
        # Get batches from dataset
        j = 0
        bar = tqdm(dataset, total= 122_000 // batch_size)
        for gray_batch, embedding_batch, lab_batch in dataset:
            with tf.GradientTape() as g_tape:
                # Project embeddings to latent dimension
                embedding_w = map_layer(tf.squeeze(embedding_batch))
            
                # Forward Prop
                pred_ab = model([tf.reshape(gray_batch, (-1, 256, 256, 1)), tf.squeeze(embedding_w)])
                
                # Calculate loss and backprop
                # l2_reg = 0.00000001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in model.trainable_variables ])
                # l2_reg += 0.00000001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in map_layer.trainable_variables ])
                loss = loss_fn(pred_ab, lab_batch[:, :, :, 1:] / 128)

            trainable_weights = (
                map_layer.trainable_weights + model.trainable_weights
            )
            gradients = g_tape.gradient(loss, trainable_weights)
            optimizer.apply_gradients(zip(gradients, trainable_weights))
                
            saved_loss.append(loss.numpy())
            # logging.info(f"Loss for batch {j}: {loss.numpy():0.3f}")
            bar.set_description(f"Loss for batch {j} => {loss.numpy():0.5f}")
            bar.update()
            j+=1
        manager.save()
        bar.close()

        # Run a validation loop at the end of each epoch
        # val_loss = 0
        # N = 0
        # for gray_batch, embedding_batch, lab_batch in validation:
            # Project embeddings to latent dimension
            # embedding_w = map_layer(tf.squeeze(embedding_batch))

            # Forward Prop
            # pred_ab = model([tf.reshape(gray_batch, (-1, 256, 256, 1)), tf.squeeze(embedding_w)])

            # Calculate loss and backprop
            # val_loss += loss_fn(pred_ab, lab_batch[:, :, :, 1:] / 128)
            # N += 1
        # logging.info(f"Validation Loss: {val_loss / N}")
        # saved_val_loss.append(val_loss / N)
        # saved_loss = np.array(saved_loss)
        # loss_epochs.append(saved_loss.mean())
        # saved_loss = []
    
    # Save loss plot
    plt.figure()
    loss = np.array(loss)
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss per batched iteration")
    plt.savefig("loss.pdf")


def main():
    # Set up logger
    logging.basicConfig(level=logging.INFO)

    # Set up logger and arguments
    args = parser.parse_args()

    # Get the pre-processed dataset
    # split_ind = int(0.9 * len(tfrecord.FILENAMES))
    # TRAINING_FILENAMES, VALID_FILENAMES = tfrecord.FILENAMES[:split_ind], tfrecord.FILENAMES[split_ind:]
    # dataset = tfrecord.load_dataset(TRAINING_FILENAMES)
    # validation = tfrecord.load_dataset(VALID_FILENAMES)
    # FILENAMES = tf.io.gfile.glob("./single_batch/laion*.tfrecords")
    dataset = tfrecord.load_dataset(tfrecord.FILENAMES)
    dataset = dataset.prefetch(buffer_size=tfrecord.AUTOTUNE)
    dataset = dataset.batch(int(args.batch_size))
    validation = None
    # validation = validation.prefetch(buffer_size=tfrecord.AUTOTUNE)
    # validation = validation.batch(int(args.batch_size))

    # Get models
    input_shape = (256, 256, 1)
    model = model_block(32, input_shape=input_shape, is_base=True, output=True, blocks=6)
    map_layer = Mapping(1, 1024)
    
    # Train
    train(dataset, validation, model, map_layer, 
          int(args.num_epochs), float(args.learning_rate), int(args.batch_size))


if __name__ == "__main__":
    main()
