# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

import cv2

# # The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
#
# # The number of images in the validation set.
# _NUM_VALIDATION = 350
#
# Seed for repeatability.
_RANDOM_SEED = 0
#
# # The number of shards per dataset split.
# _NUM_SHARDS = 5
#
# # The ratio of validation set in the data set.
# _RATIO_VALIDATION = 0.1


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  root_dir = os.path.join(dataset_dir, 'images')
  directories = []
  class_names = []
  for filename in os.listdir(root_dir):
    path = os.path.join(root_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_name, dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'Expw_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(dataset_name, split_name, filenames, class_names_to_ids, dataset_dir, num_shards):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_name, dataset_dir, split_name, shard_id, num_shards)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.GFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


# def _clean_up_temporary_files(dataset_dir):
#   """Removes temporary files used to create the dataset.
#
#   Args:
#     dataset_dir: The directory where the temporary files are stored.
#   """
#   filename = _DATA_URL.split('/')[-1]
#   filepath = os.path.join(dataset_dir, filename)
#   tf.gfile.Remove(filepath)
#
#   tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#   tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_name, dataset_dir, num_shards):
  for split_name in ['train', 'validation']:
    for shard_id in range(num_shards):
      output_filename = _get_dataset_filename(
          dataset_name, dataset_dir, split_name, shard_id, num_shards)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def _create_dir_and_copy_files(dataset_dir, label_file, resize_width, resize_height):
    f = open(label_file, 'r')
    lines = f.readlines()
    f.close()
    save_dir = dataset_dir + '/images/'
    cnt = 0
    for line in lines:
        data = line[:-1].split()
        path = data[0]
        lable = data[1]
        if not (os.path.isdir(save_dir + lable)):
            os.makedirs(os.path.join(save_dir + lable))
        image = cv2.imread(path)
        if image.shape[0] != resize_height or image.shape[1] != resize_width:
            image = cv2.resize(image, dsize=(resize_width, resize_height))
        filename = os.path.basename(path)
        save_path = save_dir + data[1] + '/' + filename
        cv2.imwrite(save_path, image)
        cnt += 1
        print('Copy\tfrom: {}\n    \tTo: {}\n'.format(path, save_path))
        print('Count: {} of {}\n'.format(cnt, len(lines)))

    return save_dir


def run(dataset_name, dataset_dir, num_shards, ratio_val, label_file, image_width, image_height):
  """Runs the conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.

  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_name, dataset_dir, num_shards):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  root_dir = _create_dir_and_copy_files(dataset_dir, label_file, image_width, image_height)

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  num_validation = int(len(photo_filenames) * ratio_val)
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[num_validation:]
  validation_filenames = photo_filenames[:num_validation]

  # First, convert the training and validation sets.
  _convert_dataset(dataset_name, 'train', training_filenames, class_names_to_ids,
                   dataset_dir, num_shards)
  _convert_dataset(dataset_name, 'validation', validation_filenames, class_names_to_ids,
                   dataset_dir, num_shards)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(root_dir)
  print('\nFinished converting the dataset!')
