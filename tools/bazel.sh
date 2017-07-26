#!/bin/bash

#bazel build //tensorflow/tensorflow/tools/graph_transforms:transform_graph 
~/WorkShop/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph="frozen.pb" \
  --out_graph="bazel.pb" \
  --inputs='input' \
  --outputs='output' \
  --transforms='
  strip_unused_nodes
    '
    #strip_unused_nodes(type=float, shape="32,256,256,3")
    #strip_unused_nodes
    #fold_constants(ignore_errors=true)
    #fold_batch_norms
    #fold_old_batch_norms
    #sort_by_execution_order
    #add_default_attributes
    ##remove_nodes(op=Identity, op=shuffle_batch)
    ##quantize_weights
    ##quantize_nodes
##  --transforms='
##    strip_unused_nodes(type=float, shape="1,299,299,3")
##    remove_nodes(op=Identity, op=CheckNumerics)
##    fold_constants(ignore_errors=true)
##    fold_batch_norms
