# -*- coding: utf-8 -*-
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

"""Evaluation for CARC-19.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import model
from carc19_class import CARC19_CLASS

IMAGE_CHANNEL=3
IMAGE_SIZE=256

def analyze_once(sess, logits, top_k, images, num_iter):
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    for step in range(int(num_iter)):
      vlogits, vtop_k, images = sess.run([logits, top_k, images])
      for j in range(len(vtop_k.values)):
        print ('Predict(%d):' % j, end='')
  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)
  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
  return vlogits, vtop_k

def prepare():
  with tf.Graph().as_default() as g:
    #image = tf.constant(0, dtype=tf.float32, shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = tf.placeholder(tf.uint8, shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), name="input")
    #image = tf.identity(image, "input")
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = tf.image.per_image_standardization(image)
    image = tf.identity(image, "output")

    with tf.Session() as sess:
      tf.train.write_graph(sess.graph, './', 'image_preproc.pb', as_text=False)


def main(argv=None):  # pylint: disable=unused-argument
  image_file = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34/image/24/cn/o_1bbig1v504226328749310003144644059.jpg'
  prepare()

if __name__ == '__main__':
  tf.app.run()
