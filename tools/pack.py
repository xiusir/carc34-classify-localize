# -*- coding: utf-8 -*-
# Preparing a TF model for usage in Android
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


MODEL_NAME = 'carc34_v0.9.0'

path='../tmp/carc34_train'
input_graph_path = '%s/graph.pbtxt' % path
checkpoint_path = '%s/model.ckpt-100000' % path
input_saver_def_path = ""
input_binary = False
input_node_names = "input"
output_node_names = "softmax_linear/softmax_linear"
#output_node_names = "softmax_linear/softmax_linear,output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True

# Freeze the graph
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

# Optimize for inference
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def,
                input_node_names.split(","),  # an array of the input node(s)
                output_node_names.split(","), # an array of the output nodes
                tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
##tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)   #

