# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Routine for decoding the CARC-19 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Process images of this size. square image: width = height = IMAGE_SIZE
IMAGE_SIZE = FLAGS.image_size
IMAGE_CHANNEL = 3

# Global constants describing the CARC-19 data set.
NUM_CLASSES = FLAGS.num_classes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.train_size
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.num_examples


def read_image(filename_queue):
  """Reads and parses examples from CARC-19 data files.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (IMAGE_SIZE)
      width: number of columns in the result (IMAGE_SIZE)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CARC-19 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # https://www.tensorflow.org/api_docs/python/tf/WholeFileReader
  reader = tf.WholeFileReader(name='image_reader')
  key, value = reader.read(filename_queue)

  # Decode jpg-formated images
  # https://www.tensorflow.org/api_docs/python/tf/image/decode_jpeg
  # A Tensor of type uint8. 3-D with shape [height, width, channels]
  image = tf.image.decode_jpeg(value, channels=IMAGE_CHANNEL)
  image = tf.cast(image, tf.float32)
  image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

  #### Convert from [depth, height, width] to [height, width, depth].
  ###image = tf.transpose(depth_major, [1, 2, 0])
  return key,image

def _generate_image_and_label_batch(image, label, key,
                                    min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch, keys = tf.train.shuffle_batch(
        [image, label, key],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch, keys = tf.train.batch(
        [image, label, key],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, tf.reshape(label_batch, [batch_size]), tf.reshape(keys, [batch_size])


def train_inputs(data_dir, batch_size=8, casefile='label_for_train.dat', distort=True, shuffle=True):
  """Construct input for CARC training using the Reader ops.

  Args:
    data_dir: Path to the CARC-19 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  case_queue = tf.train.string_input_producer([os.path.join(data_dir, casefile)], shuffle=False)
  line_reader = tf.TextLineReader()
  key, value = line_reader.read(case_queue)
  record_defaults = [[''], [''], [0], [''], [''], [''], [''], ['']] 
  path, filename, label, _,_,_,_,_  = tf.decode_csv(value, record_defaults=record_defaults, field_delim=' ')

  with open(os.path.join(data_dir, casefile), 'r') as label_items:
    filenames = [ ]
    for line in label_items:
      parts = line.strip().split(' ')
      filenames.append(os.path.join(data_dir, parts[0]+parts[1]))

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

  # Read examples from files in the filename queue.
  key, image = read_image(filename_queue)
  reshaped_image = tf.cast(image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE
  reshaped_image.set_shape([height, width, 3])

  # Image processing for training the network. Note the many random
  # distortions applied to the image.
  distorted_image = reshaped_image
  if distort:
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=64)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.1, upper=2.0)
    # se diao
    distorted_image = tf.image.random_hue(distorted_image, 0.4)
    # bao he du
    distorted_image = tf.image.random_saturation(distorted_image, 0.2, 0.9)
    # TODO 在一定范围内旋转 平面旋转、立体旋转
    # rotate
    # distorted_image = tf.image.rot90(distorted_image, k=num_rot)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)
  #float_image = distorted_image

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])

  label = tf.reshape(label, [1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.001
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CARC images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label, key, 
                                         min_queue_examples, batch_size,
                                         shuffle)

def evaluate_inputs(eval_data, data_dir, batch_size):
  """Construct input for CARC evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CARC-19 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    keys: Keys. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    label_file = 'label_for_train.dat'
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    label_file = 'label_for_test.dat'
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  return train_inputs(data_dir, batch_size=batch_size, casefile=label_file,
                      shuffle=False, distort=False)

