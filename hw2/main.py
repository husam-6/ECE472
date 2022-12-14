"""
Homework assignment 2 - Husam Almanakly

Picking a function f was difficult because our model required our output to be between
0-1. So I couldn't just use any activation function for each layer (we can for the
hidden and input layers, but ultimately at the output we need to have a scaled number between
0-1). 

To get around this I just used a sigmoid activation in the last layer / output layer, since
a sigmoid = 1 / (1 + exp(x)).

I played around with leaky_relu, elu, gelu, and tanh and eventually once my code was working, 
stuck with 2 layers being relu (output being a sigmoid as well as mentioned). I had some trouble
when I used all sigmoids because the gradient would saturate (small derivative)

It's also worth noting we aren't exactly defining the function - our Network is really learning
the function f using a bunch of weights and our SGD process to approximate it! That's sort of
the point of a neural network

"""

import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from tqdm import trange
from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))
matplotlib.style.use("classic")

THETA_LOW = np.pi
THETA_HIGH = 9 * np.pi / 2
LAMB = 0.001

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
    """Data class for spiral generated data"""
    rng: InitVar[np.random.Generator]
    num_samples: int
    sigma: float
    
    # Spiral 1
    theta: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    true: np.ndarray = field(init=False)
    

    def __post_init__(self, rng):
        self.index = np.arange(2 * self.num_samples)
        self.theta = rng.uniform(THETA_LOW, THETA_HIGH, size=(self.num_samples, 1))
        
        x1 = self.theta * np.cos(self.theta) + rng.normal(loc=0, scale=self.sigma, size=(self.num_samples, 1))
        x2 = - self.theta * np.cos(self.theta) + rng.normal(loc=0, scale=self.sigma, size=(self.num_samples, 1))
        
        y1 = - self.theta * np.sin(self.theta) + rng.normal(loc=0, scale=self.sigma, size=(self.num_samples, 1))
        y2 = self.theta * np.sin(self.theta) + rng.normal(loc=0, scale=self.sigma, size=(self.num_samples, 1))

        self.x = np.concatenate((x1, x2))
        self.y = np.concatenate((y1.flatten(), y2.flatten()))
        self.true = np.concatenate((np.zeros(self.num_samples), np.ones(self.num_samples)))

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices], self.true[choices]


# TensorFlow Example documentation used: 
# https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
def xavier_init(input_dim, output_dim, rng):
    """Initializes weight vector given in/out dimensions
    
    Obtained from TensorFlow Documentation
    """
    xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(input_dim + output_dim, tf.float32))
    weight_vals = rng.uniform(shape=(input_dim, output_dim), minval=-xavier_lim, maxval=xavier_lim)
    return weight_vals

class layer(tf.Module):
    """Layer class to build each Neural Network Layer
    
    Obtained from TensorFlow documentation
    """
    def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
    # Initialize the dimensions and activation functions
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x, rng):
        if not self.built:
            # Infer the input dimension based on first call
            self.in_dim = x.shape[1]
            # Initialize the weights and biases using Xavier scheme
            self.w = tf.Variable(xavier_init(self.in_dim, self.out_dim, rng))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim, )))
            self.built = True
        
        # Compute the forward pass, apply the layer to the data
        z = x @ self.w + self.b
        return self.activation(z)


class Model(tf.Module):
    def __init__(self, layers):
        """
        Model for a Multi Layer Perceptron to classify spiral dataset

        Also obtained from TensorFlow documentation
        """
        self.layers = layers

    def __call__(self, x, rng):
        """Instantiates instance of model - applies each layer initialized above"""
        for layer in self.layers:
            x = layer(x, rng)
        
        return tf.squeeze(x)
    
    @property
    def model(self):
        return self.layers


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

    data = Data(
        np_rng,
        num_samples,
        float(args.sigma_noise),
    )
    
    # Binary Cross Entropy loss
    def loss_func(y, y_hat):
        loss = - y * tf.math.log(y_hat) - (1 - y) * tf.math.log(1 - y_hat) + 1e-15
        return tf.reduce_mean(loss)

    # Apply model
    model = Model([
                layer(100, activation=tf.nn.relu),
                layer(50, activation=tf.nn.relu),
                layer(1, activation=tf.math.sigmoid)
    ])

    # optimizer = tf.optimizers.SGD(learning_rate=float(args.learning_rate))
    optimizer = tf.optimizers.Adam(learning_rate=float(args.learning_rate))
    bar = trange(int(args.num_iters))
    for i in bar:
        with tf.GradientTape() as tape:
            xs, ys, true = data.get_batch(np_rng, batch_size)
            coordinates = np.concatenate((tf.squeeze(xs), ys)).reshape(2, batch_size).T
            y_hat = model(coordinates, tf_rng)
            l2_reg = LAMB * tf.reduce_mean([tf.nn.l2_loss(v) for v in model.trainable_variables ])
            loss = loss_func(true, y_hat) + l2_reg

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        bar.set_description(f"Loss @ Epoch {i} => {loss.numpy():0.3f}")
        bar.refresh()


    # Plot data
    num_points = 100
    axis = np.linspace(-15, 15, num_points)
    xx, yy = np.meshgrid(axis, axis)
    
    # Reshape for easier work with Model
    coords = np.vstack([xx.ravel(), yy.ravel()]).T
    y = model(coords, tf_rng)
    output = tf.squeeze(y)
    
    # Plot real samples and predicted boundary curve at p(x) = 0.5
    plt.figure()
    plt.plot(data.x[:num_samples], data.y[:num_samples], "o", color="red")
    plt.plot(data.x[num_samples:], data.y[num_samples:], "o", color="blue")
    plt.legend(["Data 0", "Data 1"])
    plt.contourf(xx, yy, output.numpy().reshape(num_points, num_points), [0, 0.5, 1], colors=["lightcoral", "steelblue"])
    plt.title("Spiral Data")
    
    plt.tight_layout()
    plt.savefig(f"{script_path}/spiral.pdf")


if __name__ == "__main__":
    main()
