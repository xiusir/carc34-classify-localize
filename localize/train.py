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

"""A binary to train CARC-19 using a single GPU.

Accuracy:
carc19_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by carc19_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CARC-19
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import carc_flags
import model

FLAGS = tf.app.flags.FLAGS

def train():
  """Train CARC-19 for a number of steps."""
  with tf.Graph().as_default():
    tf.set_random_seed(int(time.time()))

    ##global_step = tf.contrib.framework.get_or_create_global_step()
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    global_step = tf.contrib.framework.get_or_create_global_step()
    if ckpt and ckpt.model_checkpoint_path:
      # This is only for the logger (e.g., it is not responsible for saving).
      global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      global_step_init = -1

    # Get images and labels for CARC-19.
    images, boxes = model.train_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate loss.
    loss,accuracy,absdiff,cploss = model.loss(logits, boxes)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = global_step_init
        self._start_time = time.time()
        self._saver = tf.train.Saver(max_to_keep=None)
        self._sess = None

      def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        self._sess = session

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss,accuracy,absdiff,cploss])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if (self._step+1) % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %s (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step+1, loss_value,
                               examples_per_sec, sec_per_batch))
        if (self._step+1) % FLAGS.save_frequency == 0:
          self._saver.save(self._sess, "%s-model.ckpt-%s" % (FLAGS.train_dir_save, self._step+1))

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=600,
        config=config) as mon_sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(mon_sess, ckpt.model_checkpoint_path)
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  model.maybe_download_and_extract()
  ##if tf.gfile.Exists(FLAGS.train_dir):
  ##  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
