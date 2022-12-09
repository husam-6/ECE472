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
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
GRAY_IMAGE_SIZE = (256, 256)
LAB_IMAGE_SIZE = (256, 256, 3)


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

# Read from TFRecords file
# raw_image_dataset = tf.data.TFRecordDataset('./tfrecords/laion0.tfrecords', compression_type="GZIP")

# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
# i = 0
# for image_features in parsed_image_dataset:
#     if i == 10:
#         break
#     i+=1
#     image_raw = image_features["image_raw"].numpy()
#     gray_raw = image_features["image_gray"].numpy()
#     caption = image_features["caption_emb"].numpy()
#     caption = np.frombuffer(caption, dtype=np.float32)

#     image = np.frombuffer(image_raw, dtype=np.int8).astype(np.float64)
#     image = np.resize(image, (256, 256, 3))
    
#     gray_i = np.frombuffer(gray_raw, dtype=np.int8).astype(np.float64)
#     gray_i = np.resize(gray_i, (256, 256))

#     plt.figure()
#     plt.imshow(color.lab2rgb(image), cmap="gray")
#     plt.figure()
#     plt.imshow(gray_i, cmap="gray")
#     # print(caption)


def read_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_desc)

    image_raw = example["image_raw"]
    gray_raw = example["image_gray"]
    caption = example["caption_emb"]

    caption = tf.io.decode_raw(caption, tf.float64)

    image = tf.io.decode_raw(image_raw, tf.int8)
    image = tf.reshape(image, LAB_IMAGE_SIZE)
    image = tf.cast(image, tf.float64)
    
    gray = tf.io.decode_raw(gray_raw, tf.int8)
    gray = tf.reshape(gray, GRAY_IMAGE_SIZE)
    gray = tf.cast(gray, tf.float64)

    return gray, caption, image


def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP"
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

FILENAMES = tf.io.gfile.glob("./tfrecords/laion*.tfrecords")
# split_ind = int(0.9 * len(FILENAMES))
# TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]
TRAINING_FILENAMES = FILENAMES

# TEST_FILENAMES = tf.io.gfile.glob("./tfrecords/test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
# print("Validation TFRecord Files:", len(VALID_FILENAMES))
# print("Test TFRecord Files:", len(TEST_FILENAMES))

# %%
dataset = load_dataset(TRAINING_FILENAMES)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)

# %%

def show_batch(image_batch, label_batch, caption):
    for n in range(5):
        plt.figure()
        plt.imshow(color.lab2rgb(label_batch[n].numpy()), cmap="gray")
        plt.figure()
        plt.imshow(image_batch[n], cmap="gray")

        print(caption[n])


gray_batch, embedding_batch, lab_batch = next(iter(dataset))
show_batch(gray_batch, lab_batch, embedding_batch)
plt.show()
# if __name__ == "__main__":
#     main()