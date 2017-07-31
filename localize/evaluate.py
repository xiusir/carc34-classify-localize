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

Accuracy:
carc19_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by carc19_eval.py.

Speed:
On a single Tesla K40, carc19_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CARC-19
data set, compile the program and train the model.

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

import carc_flags
import model
from carc_class import CARC19_CLASS

FLAGS = tf.app.flags.FLAGS


def analyze_once(saver, summary_writer, summary_op, keys, boxes, logits, loss, step=None):
  """Evaluate and analyze.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    if step is None:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/carc34_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
    else:
      saver.restore(sess, "%s-model.ckpt-%s" % (FLAGS.train_dir_save, step))
      global_step = step

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      step = 0
      error_count = 0
      avg_iou = 0.0
      while step < num_iter and not coord.should_stop():
        boxes1, keys1, logits1, loss1 = sess.run([boxes, keys, logits, loss])
        print("step: %s  [loss, iou, diff] = %s" % (global_step, str(loss1)))
        avg_iou += loss1[1] / num_iter
        for idx in range(len(keys1)):
          print (u" %s | %s | %s" % (str(keys1[idx]), boxes1[idx], logits1[idx]))
        step = step + 1
      print ("average_iou = %.4lf" % avg_iou)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def analyze(step=None):
  """Eval CARC-19 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CARC-19.
    eval_data = FLAGS.eval_data == 'test'
    keys, images, boxes = model.evaluate_inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images, train=False)
    loss = model.loss(logits, boxes)
    #loss = model.difference_loss(logits, boxes)
    #accuracy = model.iou(logits, boxes)

    # Calculate predictions.
    #top_k_op = tf.nn.top_k(logits, k=1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    analyze_once(saver, summary_writer, summary_op, keys, boxes, logits, loss, step=step)

def main(argv=None):  # pylint: disable=unused-argument
  model.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)

  analyze()

if __name__ == '__main__':
  tf.app.run()
