from flask import Flask, jsonify, request
import urllib.request

import sys
import time
import base64
import os
import json

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('classify_model_path', '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train.10w.20170622/frozen_custom.pb',
                           """Path to model data.""")
tf.app.flags.DEFINE_string('localize_model_path', '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_train/frozen_custom.pb',
                           """Path to model data.""")
tf.app.flags.DEFINE_integer('port', 5000,
                            """Application port.""")
tf.app.flags.DEFINE_integer('top_k', 5,
                            """Finds the k largest entries""")
tf.app.flags.DEFINE_integer('input_size', 256,
                            """Size of input image""")
tf.app.flags.DEFINE_string('input_node', 'input_image:0',
                            """Grapy input point""")
tf.app.flags.DEFINE_string('output_node', 'output:0',
                            """Grapy output point""")
def load_label_strings():
    # retrieve labels
    labels = {}
    with open('./label.dict', 'r') as f:
      for line in f:
        p = line.strip().split(' ')
        labels[p[0]] = p[1]
    print('{} labels loaded.'.format(len(labels)))
    return labels
 
def load_classify_model():
    # load model data, get top_k
    if not os.path.isfile(FLAGS.classify_model_path):
        print('No model data file found')
        sys.exit(255)
    
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.classify_model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    
    print (sess.graph)
    tf.import_graph_def(graph_def, name='')
    print (sess.graph)
    input_node = sess.graph.get_tensor_by_name(FLAGS.input_node)
    output_node = sess.graph.get_tensor_by_name(FLAGS.output_node)
    print (input_node)
    print (output_node)
    top_values, top_indices = tf.nn.top_k(output_node, k=FLAGS.top_k)
    return sess, top_values, top_indices
    
   
    ###TODO DEBUG
    ###print (time.time())
    ###data = open('images/o_1bc4q9nfb1181222690274524715012536.jpg', 'rb').read(100000)
    ###ops = [top_values, top_indices]
    ###values, indices = sess.run(ops, feed_dict={FLAGS.input_node: data})
    ###top_k = []
    ###for i in range(FLAGS.top_k):
    ###    top_k.append({
    ###        'label': labels.get(str(indices.flatten().tolist()[i]), {}),
    ###        'value': values.flatten().tolist()[i],
    ###    })
    ###print ({'top': top_k})
    ###print (time.time())
    #####sys.exit(0)

def load_localize_model():
    # load model data, get top_k
    if not os.path.isfile(FLAGS.localize_model_path):
        print('No model data file found')
        sys.exit(255)
    
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.localize_model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    
    print (sess.graph)
    tf.import_graph_def(graph_def, name='')
    print (sess.graph)
    input_node = sess.graph.get_tensor_by_name(FLAGS.input_node)
    output_node = sess.graph.get_tensor_by_name(FLAGS.output_node)
    output_node = tf.reshape(output_node, [4])
    print (input_node)
    print (output_node)
    return sess, output_node
    
    ####TODO DEBUG
    ###print (time.time())
    ###data = open('images/o_1bc4q9nfb1181222690274524715012536.jpg', 'rb').read(100000)
    ###ops = [output_node]
    ###box = sess.run(ops, feed_dict={FLAGS.input_node: data})
    ###print ({'boxpos': box})
    ###print (time.time())
    #####sys.exit(0)

labels = load_label_strings()
classify_sess, top_values, top_indices = load_classify_model()
#localize_sess, boxpos = load_localize_model()

# Flask setup
app = Flask(__name__)
app.debug = True


@app.route('/labels')
def label():
    return jsonify(labels=labels)


@app.route('/classify', methods=['POST'])
def api_classify():
    results = []
    ops = [top_values, top_indices]
    ###for image in request.files.getlist("images[]"):
    ###    app.logger.info("%s %s " % (image, type(image)))
    ###return jsonify(results={'ok':len(request.files.getlist("images[]"))})
    for image in request.files.getlist("images[]"):
        data = image.read()
        #app.logger.info("%s %s" % (len(data), image.content_length))
        values, indices = classify_sess.run(ops, feed_dict={FLAGS.input_node: data})
        top_k = []
        for i in range(FLAGS.top_k):
            top_k.append({
                'label': labels.get(str(indices.flatten().tolist()[i]), {}),
                'value': values.flatten().tolist()[i],
            })
        results.append({'top': top_k})
    return jsonify(results=results)

@app.route('/localize', methods=['POST'])
def api_localize():
    results = []
    ops = boxpos
    for image in request.files.getlist("images[]"):
        data = image.read()
        box = localize_sess.run(ops, feed_dict={FLAGS.input_node: data})
        app.logger.info("%s %s " % (box, type(box)))
        results.append({'boxpos': box.tolist()})
    return jsonify(results=results)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port) 
