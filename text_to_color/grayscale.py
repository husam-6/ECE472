# %% Libraries
import matplotlib.pyplot as plt
import numpy as np

# %% Script to convert 'true' colored images to grayscale

# Convert to grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def main():
    images = np.load("images.npy")
    gray_images = None
    for i in range(images.shape[0]):
        img = images[i, :, :]
        gray = rgb2gray(img)
        if i == 0:
            gray_images = gray
            continue
        
        gray_images = np.hstack((gray_images, gray))

    with open("gray_images.npy", "wb") as f:
        np.save(f, gray_images)

if __name__ == "__main__":
    main()
