#!/bin/bash

MODEL="../tmp/carc34_train.20170621_160315"
MODEL="../tmp/carc34_train"

#OUTPUT_NODES="output,conv1/weights/ExponentialMovingAverage,conv1/biases/ExponentialMovingAverage,conv2/weights/ExponentialMovingAverage,conv2/biases/ExponentialMovingAverage,conv3/weights/ExponentialMovingAverage,conv3/biases/ExponentialMovingAverage,conv5/weights/ExponentialMovingAverage,conv5/biases/ExponentialMovingAverage,local6/weights/ExponentialMovingAverage,local6/biases/ExponentialMovingAverage,softmax_linear/weights/ExponentialMovingAverage,softmax_linear/biases/ExponentialMovingAverage"
OUTPUT_NODES="output"

##/home/xiusir/WorkShop/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
##  --input_graph=./image_preproc.pbtxt \
##  --output_node_names=output  \
##  --output_graph=image_preproc.pb

#optional arguments:
#  --input_graph INPUT_GRAPH TensorFlow 'GraphDef' file to load.
#  --input_saver INPUT_SAVER TensorFlow saver file to load.
#  --input_checkpoint INPUT_CHECKPOINT TensorFlow variables file to load.
#  --output_graph OUTPUT_GRAPH Output 'GraphDef' file name.
#  --input_binary [INPUT_BINARY] Whether the input files are in binary format.
#  --output_node_names OUTPUT_NODE_NAMES The name of the output nodes, comma separated.
#  --restore_op_name RESTORE_OP_NAME The name of the master restore operator.
#  --filename_tensor_name FILENAME_TENSOR_NAME The name of the tensor holding the save path.
#  --clear_devices [CLEAR_DEVICES] Whether to remove device specifications.
#  --initializer_nodes INITIALIZER_NODES comma separated list of initializer nodes to run before freezing.
#  --variable_names_blacklist VARIABLE_NAMES_BLACKLIST comma separated list of variables to skip converting to constants

/home/xiusir/WorkShop/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph="$MODEL"/graph.pbtxt \
  --input_checkpoint="$MODEL"/model.ckpt-100000 \
  --output_graph=frozen.pb \
  --output_node_names=${OUTPUT_NODES}  \
  --restore_op_name=save/restore_all \
  --clear_devices
  #--initializer_nodes="" \
  

/home/xiusir/WorkShop/tensorflow/bazel-bin/tensorflow/python/tools/optimize_for_inference \
  --input=./frozen.pb \
  --output=./inference.pb \
  --input_names=input \
  --output_names=${OUTPUT_NODES} \
  --frozen_graph=True

/home/xiusir/WorkShop/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph="frozen.pb" \
  --out_graph="bazel.pb" \
  --inputs="input" \
  --outputs="${OUTPUT_NODES}" \
  --transforms='
  strip_unused_nodes
    '
 
