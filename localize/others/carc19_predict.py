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
from tensorflow.python.platform import gfile

import model
from carc19_class import CARC19_CLASS

IMAGE_CHANNEL=3
IMAGE_SIZE=256

def analyze_once(sess, logits, top_k, images, image, image1, num_iter):
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    for step in range(int(num_iter)):
      vlogits, vtop_k, images,image,image1 = sess.run([logits, top_k, images, image, image1])
      for j in range(len(vtop_k.values)):
        print ('Predict(%d):' % j, end='')
        for i in range(len(vtop_k.values[j])):
          print (' %d:%s:%.5lf' % (vtop_k.indices[j][i], CARC19_CLASS[int(vtop_k.indices[j][i])], vtop_k.values[j][i]), end = '')
        print ()
        with open('image.dat', 'w') as out:
          for i in range(0,256):
            for j in range(0,256):
               out.write(str(image[i][j])+"\n")
  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)
  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
  return vlogits, vtop_k


def predict(imagefiles):
  """Eval CARC-19 for a number of steps."""
  #model_path = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train/model.ckpt-9801'
  model_path = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/release/model.ckpt-100000'
  model_path = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train/model.ckpt-100000'
  #model_path = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/release/bazel.pb'

  with tf.Graph().as_default() as g:
    filename_queue = tf.train.string_input_producer(imagefiles)
    reader = tf.WholeFileReader(name='image_reader')
    key, value= reader.read(filename_queue)
    #image = tf.placeholder(tf.float32, shape=(256, 256, 3), name='image')
    print (value)
    image = tf.image.decode_jpeg(value, channels=IMAGE_CHANNEL)
    image0 = tf.cast(image, tf.float32)
    image0 = tf.reshape(image0, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image1 = tf.image.per_image_standardization(image0)
    images = tf.reshape(image1, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    #images = tf.reshape(image0, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    ##images = tf.train.batch(
    ##    [image1],
    ##    batch_size=2,
    ##    num_threads=1,
    ##    capacity=128)
    #images, labels, keys = carc19.evaluate_inputs(eval_data=True)
    logits = carc19.inference(images)
    top_k = tf.nn.top_k(logits, k=10)

    # Precision: 97%
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(carc19.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    for k in variables_to_restore.keys():
      print (k,variables_to_restore[k])
    saver = tf.train.Saver(variables_to_restore)

    # Precision: 84%
    #saver = tf.train.Saver()

    #model_path = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train/model.ckpt-9801'
    with tf.Session() as sess:
      saver.restore(sess, model_path)
      vlogits, vtop_k = analyze_once(sess, logits, top_k, images, image,image1,math.ceil(len(imagefiles)/32))
      ##for j in range(len(vtop_k.values)):
      ##  print ('Predict(%d):' % j, end='')
      ##  for i in range(len(vtop_k.values[j])):
      ##    print (' %s:%.3lf' % (CARC19_CLASS[int(vtop_k.indices[j][i])], vtop_k.values[j][i]), end = '')
      ##  print ()



def main(argv=None):  # pylint: disable=unused-argument
  image_file = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_200/image/21/cn/o_1b37av55l2761979814439221740660.jpg'
  predict([image_file for i in range(1)])

if __name__ == '__main__':
  tf.app.run()
