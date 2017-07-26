from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MODEL_OUTPUT_TENSOR_NAME = 'output'


with tf.Graph().as_default() as graph:
    model_filename = os.path.join('.', 'bazel_carc34_v0.9.0.pb')
    model_filename = os.path.join('.', 'image_preproc.pb')
    model_filename = os.path.join('.', 'bazel.pb')
    model_filename = os.path.join('.', 'frozen_custom.pb')
    model_filename = os.path.join('../../MySecondModel/tmp/carc34_train.10w.20170622', 'frozen_custom.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      for n in graph_def.node:
        print (n.name)
      output_tensor = tf.import_graph_def(graph_def, name='', return_elements=[MODEL_OUTPUT_TENSOR_NAME])
      print ("Found in frozen")

    #print ("=============")
    #model_filename = os.path.join('.', 'bazel_carc34_v0.9.0.pb')
    #with gfile.FastGFile(model_filename, 'rb') as f:
    #  graph_def = tf.GraphDef()
    #  graph_def.ParseFromString(f.read())
    #  for n in graph_def.node:
    #    print (n.name)
    #  ##output_tensor = tf.import_graph_def(graph_def, name='', return_elements=[MODEL_OUTPUT_TENSOR_NAME])
    #  ##print ("Found in optimized")
