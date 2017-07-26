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

"""Build .pb file for trained model in tmp/carc34_train
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
import sys

import carc_flags
import model

#slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

MODEL = FLAGS.train_dir
#MODEL = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train.10w.20170622'
checkpoint_file = tf.train.latest_checkpoint(MODEL)
EXPORT_FOR_MOBILE = False

with tf.Graph().as_default() as graph:
    if EXPORT_FOR_MOBILE:
      image = tf.placeholder(shape=[FLAGS.image_size, FLAGS.image_size, 3], dtype=tf.float32, name = FLAGS.input_node)
      image0 = tf.cast(image, tf.float32)
      images = tf.image.per_image_standardization(image0)
    else:
      data = tf.placeholder(dtype=tf.string, name=FLAGS.input_node)
      image = tf.image.decode_jpeg(data, channels=3)
      image = tf.reshape(image, [256, 256, 3])
      image0 = tf.cast(image, tf.float32)
      images = tf.image.per_image_standardization(image0)

    #with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits = model.inference(images)
    variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    #Setup graph def
    input_graph_def = graph.as_graph_def()
    output_node_names = FLAGS.output_node
    output_graph_name = MODEL + '/' + FLAGS.export_model_file

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)

        #Exporting the graph
        print ("Exporting graph...")
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(","))

        with tf.gfile.GFile(output_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())

