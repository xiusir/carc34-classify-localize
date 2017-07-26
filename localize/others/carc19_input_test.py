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

"""Tests for carc19 input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import carc19_input


def testSimple():
  #with tf.Graph().as_default() as g:
  images, boxes = carc19_input.train_inputs('../tmp/carc34', 32)

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    ##q = tf.FIFOQueue(99, tf.string, shapes=())
    ##for filename in filenames:
    ##  q.enqueue(filename).run()
    ##q.close().run()
    ##coord = tf.train.Coordinator()
    ##threads = tf.train.start_queue_runners(coord=coord)

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

      print ("A=====")

      step = 0
      while step < 1 and not coord.should_stop():
        a, b = sess.run([images, boxes])
        ##print ("---------")
        print (a.shape)
        print (b.shape)
        ##print ("---------")
        step = step + 1

    except Exception as e:  # pylint: disable=broad-except
      print ("B=====")
      raise e
      coord.request_stop(e)

    print ("C=====")
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)



def main(argv=None):  # pylint: disable=unused-argument
  testSimple()

if __name__ == "__main__":
  #tf.test.main()
  tf.app.run()
