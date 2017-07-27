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

"""Builds the CARC-19 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import carc_flags
import data_pipeline

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# TODO tf_home must be changed to be your tensorflow working dir. Structure like:
# ex. ${tf_home}/tmp/carc34/{label_for_test.dat,label_for_train.dat,image/...}
# tf_home = /home/work/tensorflow
# mkdir -p /home/work/tensorflow/tmp
# mv small4w /home/work/tensorflow/tmp/carc34   
# OR mv big34w /home/work/tensorflow/tmp/carc34
# cd ???/carc34/model

# Global constants describing the CARC-19 data set.
IMAGE_SIZE = FLAGS.image_size
NUM_CLASSES = FLAGS.num_classes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.train_size
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.num_examples


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 250.0      # Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.95  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
KEEP_PROB = 0.5

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'



def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, use_cpu=True):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  if use_cpu:
    with tf.device('/cpu:0'):
      dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
      #var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
      var = tf.Variable(initializer(shape, dtype=dtype), name=name)
  else:
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.Variable(initializer(shape, dtype=dtype), name=name)

  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def train_inputs():
  """Construct distorted input for CARC training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  keys, images, boxes = data_pipeline.train_inputs(data_dir=data_dir,
                                             batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    boxes = tf.cast(labels, tf.float16)
  return images, boxes


def evaluate_inputs(eval_data):
  """Construct input for CARC evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  keys, images, labels = data_pipeline.evaluate_inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return keys, images, labels


def inference(images, train=True):
  """Build the CARC-19 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  #float_image = tf.image.per_image_standardization(distorted_image)
  #print (images,images.shape)
  images = tf.identity(images, 'input')
  images = tf.reshape(images, [-1, 256, 256, 3])
  #images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[11, 11, 3, 32],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1],
                         padding='SAME', name='pool1')
  if train:
    pool1 = tf.nn.dropout(pool1, KEEP_PROB)

  # conv2 - 64,64,32
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3 32,32,96
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv3)
  # pool3
  pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # conv4 16,16,128
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv4)

  # conv5 16,16,64
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 96],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv5)

  # conv6 16,16,64
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 64],
                                         stddev=5e-2,
                                         wd=0.000001)
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv6)

  conv_out =  conv6

  # local6
  with tf.variable_scope('local6') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    ##reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
    ##dim = reshape.get_shape()[1].value
    ##weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
    dim0 = conv_out.get_shape()[0].value
    dim1 = conv_out.get_shape()[1].value
    dim2 = conv_out.get_shape()[2].value
    dim3 = conv_out.get_shape()[3].value
    reshape = tf.reshape(conv_out, [-1, dim1*dim2*dim3])
    dim_in = dim1*dim2*dim3
    weights = _variable_with_weight_decay('weights', shape=[dim_in, 512],
                                          stddev=5e-2, wd=0.000001)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local6)

  # local7
  with tf.variable_scope('local7') as scope:
    weights = _variable_with_weight_decay('weights', shape=[512, 512],
                                          stddev=5e-2, wd=0.000001)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
    _activation_summary(local7)

  if train:
    local7 = tf.nn.dropout(local7, KEEP_PROB)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, 4],
                                          stddev=1/512.0, wd=0.000001)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  softmax_linear = tf.identity(softmax_linear, 'output')
  return softmax_linear

  ### conv6 1,1,4096
  ##with tf.variable_scope('conv6') as scope:
  ##  kernel = _variable_with_weight_decay('weights',
  ##                                       shape=[1, 1, 4096, 1024],
  ##                                       stddev=5e-2,
  ##                                       wd=0.000001)
  ##  conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='VALID')
  ##  biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
  ##  pre_activation = tf.nn.bias_add(conv, biases)
  ##  conv6 = tf.nn.relu(pre_activation, name=scope.name)
  ##  _activation_summary(conv6)

  ### conv7 1,1,1024
  ##with tf.variable_scope('conv7') as scope:
  ##  kernel = _variable_with_weight_decay('weights',
  ##                                       shape=[1, 1, 1024, 4],
  ##                                       stddev=5e-2,
  ##                                       wd=0.000001)
  ##  conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='VALID')
  ##  biases = _variable_on_cpu('biases', [4], tf.constant_initializer(0.1))
  ##  pre_activation = tf.nn.bias_add(conv, biases)
  ##  conv7 = tf.nn.relu(pre_activation, name=scope.name)
  ##  _activation_summary(conv7)

  ### ... [1,1,4]
  ##logits = tf.reshape(conv7, [-1, 4], name='output')

  ##return logits

def loss(logits, boxes):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference(). [batch_size, 4]
    boxes: position of objects [batch_size, 4]
  Returns:
    Loss tensor of type float.
  """
  #### Calculate the average cross entropy loss across the batch.
  ###cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  ###    labels=labels, logits=logits, name='cross_entropy_per_example')
  ###cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  accuracy = iou(logits, boxes)
  #absdiff = l1_loss(logits, boxes)
  absdiff = wh_loss(logits, boxes)
  outside,outside_ratio = outside_loss(logits, boxes)
  cploss = center_loss(logits, boxes)

  #diff = - tf.log(accuracy + 1e-3) + absdiff + tf.exp(outside) -1
  #diff = - tf.log(accuracy + 5e-2) + absdiff + outside + tf.log(outside_ratio)
  #diff = - tf.log(accuracy + 1e-12) + tf.log(absdiff) + 5 * tf.log(cploss) + 6
  #diff = - tf.log(accuracy + 1e-12) + tf.log(absdiff + 1e-12) + 5 * tf.log(cploss + 1e-12)
  diff = - tf.log(accuracy + 1e-12) + tf.log(absdiff + 1e-12) + 5 * tf.log(cploss + 1e-12) + 6

  tf.add_to_collection('losses', diff)
  tf.summary.scalar('accuracy', accuracy) 

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy, absdiff, cploss

def euclidean_loss(x1, x2):
  #l2_loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), 1)))
  l2_loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), 1)))
  return l2_loss

def center_loss(predict, expect):
  delta = tf.subtract(predict, expect)
  top,bottom,left,right = tf.split(delta, [1,1,1,1], 1)
  distance = tf.multiply(tf.sqrt(tf.square(top+bottom) + tf.square(left+right)), 0.5)
  loss = tf.reduce_sum(distance)
  return loss

def wh_loss(predict, expect):
  delta = tf.subtract(predict, expect)
  top,bottom,left,right = tf.split(delta, [1,1,1,1], 1)
  loss = tf.reduce_sum(tf.abs(right-left) + tf.abs(bottom-top))
  return loss

def iou(predict, expect):
  top1,bottom1,left1,right1 = tf.split(predict, [1,1,1,1], 1)
  top2,bottom2,left2,right2 = tf.split(expect, [1,1,1,1], 1)

  topu = tf.maximum(top1, top2)
  leftu = tf.maximum(left1, left2)
  bottomu = tf.minimum(bottom1, bottom2)
  rightu = tf.minimum(right1, right2)

  heightu = tf.nn.relu(tf.subtract(bottomu, topu))
  widthu = tf.nn.relu(tf.subtract(rightu, leftu))
  areau = tf.multiply(heightu, widthu)

  area1 = tf.multiply(tf.nn.relu(tf.subtract(bottom1, top1)), tf.nn.relu(tf.subtract(right1, left1)))
  area2 = tf.multiply(tf.subtract(bottom2, top2), tf.subtract(right2, left2))

  iou = tf.reduce_mean(tf.div(areau, tf.subtract(tf.add(area1,area2),areau)))
  return iou

def l1_loss(x1, x2):
  #diff_loss = tf.reduce_mean(tf.reduce_max(tf.abs(tf.subtract(x1, x2)), axis=1))
  #diff_loss = tf.reduce_mean(tf.abs(tf.subtract(x1, x2)))
  diff_loss = tf.reduce_sum(tf.abs(tf.subtract(x1, x2)))
  return diff_loss

def l2_loss(x1, x2):
  #diff_loss = tf.reduce_mean(tf.reduce_max(tf.abs(tf.subtract(x1, x2)), axis=1))
  diff_loss = tf.reduce_sum(tf.square(tf.subtract(x1, x2)))
  return diff_loss

def outside_loss(x1, x2):
  loss = tf.nn.relu(
             tf.multiply(
               tf.subtract(x1, x2), [-1, 1, -1, 1]
               )
         )
  loss_sum = tf.reduce_sum(loss)
  loss_cnt = tf.count_nonzero(loss, dtype=tf.float32)
  return tf.div(loss_sum, loss_cnt+1e-3), loss_cnt / (FLAGS.batch_size*4)
  #return tf.reduce_mean(loss)

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CARC-19 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CARC-19 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  pass
  ##DATA_URL = 'http://www.cs.toronto.edu/~kriz/carc-19-binary.tar.gz'
  ##dest_directory = FLAGS.data_dir
  ##if not os.path.exists(dest_directory):
  ##  os.makedirs(dest_directory)
  ##filename = DATA_URL.split('/')[-1]
  ##filepath = os.path.join(dest_directory, filename)
  ##if not os.path.exists(filepath):
  ##  def _progress(count, block_size, total_size):
  ##    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
  ##        float(count * block_size) / float(total_size) * 100.0))
  ##    sys.stdout.flush()
  ##  filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
  ##  print()
  ##  statinfo = os.stat(filepath)
  ##  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  ##extracted_dir_path = os.path.join(dest_directory, 'carc-19-batches-bin')
  ##if not os.path.exists(extracted_dir_path):
  ##  tarfile.open(filepath, 'r:gz').extractall(dest_directory)