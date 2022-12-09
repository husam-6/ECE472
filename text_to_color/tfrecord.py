""" Example script to be used to save and read TFRecord files"""


# %% libraries

from transformers import T5Model, T5Tokenizer 
import numpy as np
from os.path import exists
import logging
import time
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf

path_to_file = f"./tmp/caption15.txt"
path_to_img = f"./tmp/img15.jpeg"
file_exists = exists(path_to_file)
rgb_exists = exists(path_to_img)
if not file_exists or not rgb_exists:
    print("Image doesnt exist")
    raise AssertionError
with open(path_to_file) as f:
    lines = f.readlines()

# Get Images
rgb = resize(io.imread(path_to_img), (256, 256))
gray = color.rgb2gray(rgb)
labrgb = color.rgb2lab(rgb)

model = T5Model.from_pretrained("t5-small")
tok = T5Tokenizer.from_pretrained("t5-small")

enc = tok(lines, max_length=1024, truncation=True, return_tensors="pt")

output = model.encoder(
            input_ids=enc["input_ids"], 
            attention_mask=enc["attention_mask"], 
            return_dict=True)

# get the final hidden states
emb = output.last_hidden_state.detach().numpy().squeeze()
fixed_emb = emb.mean(axis=0)

# %%
# plt.imshow(color.lab2rgb(labrgb), cmap="gray")
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# %% Write to TFRecords file for an image

# Create a dictionary with features that may be relevant.
def image_example(image_arr):
    image_string = image_arr.tobytes()
    image_shape = image_arr.shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    tf_example = image_example(labrgb)
    writer.write(tf_example.SerializeToString())


# %% Read from TFRecords file

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
for image_features in parsed_image_dataset:
    image_raw = image_features["image_raw"].numpy()


# image_raw == labrgb.tobytes()
image = np.frombuffer(image_raw, dtype=np.float64)
image = np.resize(image, (256, 256, 3))
