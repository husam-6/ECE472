#!/bin/env python3.8

"""
Homework assignment 1 - Husam Almanakly
Modelled after linear regression example by Chris Curro

Worked with Ali Ghuman and Michael Bentivegna on __call__ function in Model class
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))

# Constants to hold the max and min x values from our data...
UPPER_VAL = 1
LOWER_VAL = 0


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_basis: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(LOWER_VAL, UPPER_VAL, size=(self.num_samples, 1))
        
        # True 'y' or output values - y = sin(2*π*x)
        clean_y = np.sin(2 * np.pi * self.x)
        
        # Experimental 'y' - y = sin(2*π*x) + e
        self.y = clean_y + rng.normal(loc=0, scale=0.1, size=(self.num_samples, 1))

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_basis", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logginwg level to debug")


class Model(tf.Module):
    def __init__(self, rng, num_basis):
        """
        A regression model to estimate a sinewave using Gaussians, weights, and a bias term
        """
        self.num_basis = num_basis
        self.b = tf.Variable(tf.zeros(shape=[1, 1]), name="bias")
        self.w = tf.Variable(rng.normal(shape=[self.num_basis, 1]), name="weights")
        self.mus = tf.Variable(rng.normal(shape=[self.num_basis, 1]), name="means")
        self.sigmas = tf.Variable(rng.normal(shape=[self.num_basis, 1]), name="sigmas")

        # NOTE: Not sure if we need different dimensions for w and mu/sigma

    def __call__(self, x):
        gaussians = tf.transpose(self.w) *  \
                    tf.math.exp(-((x - tf.transpose(self.mus))**2 / \
                    (tf.transpose(self.sigmas) ** 2)))
        return tf.squeeze(tf.reduce_sum(gaussians, 1) + self.b)


def main(a):
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    data = Data(
        np_rng,
        FLAGS.num_basis,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(tf_rng, FLAGS.num_basis)

    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()


    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    ax[0].set_title("Sinewave Regression")
    ax[0].set_xlabel("x")
    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    # Plot sampled points with gaussian noise and estimated sinewave
    xs = np.linspace(np.amin(model.mus) * 3, np.amax(model.mus)*1.5, 1000)
    xs = xs[:, np.newaxis]
    yhat = model(xs)
    ax[0].plot(xs, np.squeeze(yhat), "--", color="purple", linewidth=1.5, zorder=2)
    ax[0].plot(np.squeeze(data.x), data.y, "o", color="pink", zorder=1)
    ax[0].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    ax[0].set_xlim(LOWER_VAL, UPPER_VAL)

    # Plot true sinewave
    true_y = np.sin(2*np.pi*xs)
    ax[0].plot(xs, true_y, color="teal", zorder=0)
    ax[0].legend(["Estimated", "Data", "True"])

    ax[1].set_title("Basis Functions")
    ax[1].set_xlabel("x")
    ax[1].set_xlim(np.amin(model.mus) * 3, np.amax(model.mus) * 1.3)
    h = ax[1].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    # Plot Gaussians
    gaussians = np.zeros((1000, model.num_basis))
    for i in range(model.num_basis):
        gaussians[:, i] = np.exp(-(xs.T - model.mus[i]) ** 2 / (model.sigmas[i] ** 2))
    ax[1].plot(xs, gaussians)

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")


if __name__ == "__main__":
    app.run(main)
