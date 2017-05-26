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


class CARC19InputTest(tf.test.TestCase):

  def testSimple(self):
    labels = [0, 0, 0, 0, 0, 0]
    filenames = [
            '../carc19/image/0/bj/o_1ba6n66fk5483908542219540518766.jpg',
            '../carc19/image/0/bj/o_1bacfcseu4090956218391913317938760.jpg',
            '../carc19/image/0/bj/o_1bajj5rq6123241337744585793143383.jpg',
            ]

    with self.test_session() as sess:
      q = tf.FIFOQueue(99, tf.string, shapes=())
      for filename in filenames:
        q.enqueue(filename).run()
      q.close().run()
      result = carc19_input.read_carc19(q)

      for i in range(3):
        key, label, uint8image = sess.run([
            result.key, result.label, result.uint8image])
        self.assertEqual(filenames[i], tf.compat.as_text(key))
        self.assertEqual(labels[i], label)
        #self.assertAllEqual(expected[i], uint8image)
        print (key)
        print (label)
        print (uint8image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([result.key, result.uint8image])


if __name__ == "__main__":
  tf.test.main()
