# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

"""Basic path containing working directories, """
"""as data_dir, eval_dir, checkpoint_dir, train_dir"""
tf_home = '/home/xiusir/WorkShop/TensorFlow/MySecondModel'

############################
### basic parameters
############################
tf.app.flags.DEFINE_string('tf_home', tf_home,
                           """Basic path containing working directories, """
                           """as data_dir, eval_dir, checkpoint_dir, train_dir""")
tf.app.flags.DEFINE_integer('image_size', 256, """256x256""")
tf.app.flags.DEFINE_integer('num_classes', 34, """number of classes""")
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch during train and evaluate.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('data_dir', '%s/tmp/carc34/' % tf_home,
                           """Path to the CARC-34 data directory.""")

############################
### training parameters
############################
tf.app.flags.DEFINE_integer('train_size', 25427, """number of examples for training""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_dir', '%s/tmp/carc34_train' % tf_home,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir_save', '%s/tmp/carc34_train/save' % tf_home,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('save_frequency', 20000,
                            """How often to checkpoint to the disk.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """The decay to use for the moving average.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 50,
                            """The decay to use for the moving average.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                            """Initial learning rate.""")

############################
### evaluation parameters
############################
tf.app.flags.DEFINE_integer('num_examples', 2836,  #2836
                            """number of examples for evaluating""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('eval_dir', '%s/tmp/carc34_eval' % tf_home,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '%s/tmp/carc34_train' % tf_home,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30*1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


############################
### build parameters
############################
tf.app.flags.DEFINE_string('input_node', 'input_image',
                           """Input node for exported model""")
tf.app.flags.DEFINE_string('output_node', 'output',
                           """Output node for exported model""")
tf.app.flags.DEFINE_string('export_model_file', 'frozen_custom.pb',
                           """File name for exported model""")
