from absl import flags
from absl import app
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 1, "Number of samples in record")
flags.DEFINE_integer("random_seed", 31415, "Random seed")

def main(a):
    x = np.random.uniform(0,4,FLAGS.num_samples)
    clean_y = np.sin(2 * np.pi * x)
        
    x_axis = np.linspace(0, 4, 1000)
    # Experimental 'y' - y = sin(2*π*x) + e
    y = clean_y + np.random.default_rng(FLAGS.random_seed).normal(loc=0, scale=0.1, size=FLAGS.num_samples)
    # y = clean_y + np.random.normal(loc=0, scale=0.1, size=FLAGS.num_samples)

    x, mu = np.meshgrid(np.linspace(0,2*np.pi,1000),np.linspace(0,2*np.pi,5))
    gaussian = (1/np.sqrt(2*np.pi))*np.exp(-(x-mu) ** 2)
    xs = np.linspace(0,2*np.pi,1000)
    
    ms = np.linspace(0, 2, 6)
    sigs = np.ones(6) * 0.2
    
    gaussians = np.zeros((1000, 6))
    for i in range(6):
        gaussians[:, i] = np.exp(-(xs.T - ms[i]) ** 2 / (sigs[i] ** 2))

    plt.plot(xs, gaussian.T)
    plt.show()

if __name__ == "__main__":
    app.run(main)