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

# Create a dictionary describing the features.
feature_desc = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'image_gray': tf.io.FixedLenFeature([], tf.string),
    'caption_emb': tf.io.FixedLenFeature([], tf.string),
}


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Write to TFRecords file for an image
# Create a dictionary with features that may be relevant.
def image_example(image_arr, gray_arr, caption_emb):
    image_string = image_arr.tobytes()
    gray_string = gray_arr.tobytes()
    emb_string = caption_emb.tobytes()

    feature = {
        'image_raw': _bytes_feature(image_string),
        'image_gray': _bytes_feature(gray_string),
        'caption_emb': _bytes_feature(emb_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_desc)


def main():
    # Read in example files
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

    # Write to tfrecords file 
    record_file = 'images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        tf_example = image_example(labrgb, gray, fixed_emb)
        writer.write(tf_example.SerializeToString())

    # Read from TFRecords file
    raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    for image_features in parsed_image_dataset:
        image_raw = image_features["image_raw"].numpy()
        gray_raw = image_features["image_gray"].numpy()
        caption = image_features["caption_emb"].numpy()


    caption = np.frombuffer(caption, dtype=np.float32)

    image = np.frombuffer(image_raw, dtype=np.float64)
    image = np.resize(image, (256, 256, 3))

    gray_i = np.frombuffer(gray_raw, dtype=np.float64)
    gray_i = np.resize(gray, (256, 256))

    plt.imshow(image, cmap="gray")
    plt.figure()
    plt.imshow(gray_i, cmap="gray")

    assert (gray_i == gray).all()
    assert (image == labrgb).all()
    assert (caption == fixed_emb).all()



if __name__ == "__main__":
    main()