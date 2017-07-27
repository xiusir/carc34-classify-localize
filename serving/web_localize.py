from flask import Flask, jsonify, request, make_response
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
tf.app.flags.DEFINE_integer('port', 5001,
                            """Application port.""")
tf.app.flags.DEFINE_integer('top_k', 5,
                            """Finds the k largest entries""")
tf.app.flags.DEFINE_integer('input_size', 256,
                            """Size of input image""")
tf.app.flags.DEFINE_string('input_node', 'input_image:0',
                            """Grapy input point""")
tf.app.flags.DEFINE_string('output_node', 'output:0',
                            """Grapy output point""")


def resize_image(data):
    import tempfile
    import math
    import cv2

    temp = tempfile.NamedTemporaryFile(suffix='.jpg')
    temp.write(data)

    imagePath = temp.name
    image = cv2.imread(imagePath)

    # CONSTANT 用颜色填充
    BLACK = [0,0,0]
    # top,bottom,left,right
    width = image.shape[1]
    height = image.shape[0]

    if width == 256 and height == 256:
      return data

    padding = (width - height) * 0.5
    padding1 = math.ceil(padding)
    padding2 = math.floor(padding)
    if padding < 0:
      return data

    image = cv2.copyMakeBorder(image,padding1,padding2,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    image = cv2.resize(image, (256, 256) , interpolation = cv2.INTER_AREA)

    cv2.imwrite(imagePath, image)
    temp.seek(0)
    return temp.read()

###TODO DEBUG
###ff = 'images/o_1blp9bfc621457231430079257828839.jpg'
###xx = open(ff, 'rb').read()
###yy = resize_image(xx)
###print (len(xx), len(yy))
###exit(0)
 

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
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
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

def curl(url):
    import urllib.request
    resp=urllib.request.urlopen(url)
    return resp.read()

def load_localize_model():
    # load model data, get top_k
    if not os.path.isfile(FLAGS.localize_model_path):
        print('No model data file found')
        sys.exit(255)
    
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
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

##labels = load_label_strings()
##classify_sess, top_values, top_indices = load_classify_model()
localize_sess, boxpos = load_localize_model()

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
    app.logger.info("%s %s %s " % (request, request.files, request.form))
    images = []
    for image in request.files.getlist("images[]"):
        data = image.read()
        data = resize_image(data)
        images.append(data)
    for url in request.form.getlist("images[]"):
        data = curl(url)
        data = resize_image(data)
        images.append(data)
    for data in images:
        values, indices = classify_sess.run(ops, feed_dict={FLAGS.input_node: data})
        top_k = []
        for i in range(FLAGS.top_k):
            top_k.append({
                'label': labels.get(str(indices.flatten().tolist()[i]), {}),
                'value': values.flatten().tolist()[i],
            })
        results.append({'label': top_k})
    response = jsonify(results=results)
    response = make_response(jsonify(results=results))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/localize', methods=['POST'])
def api_localize():
    results = []
    ops = boxpos

    images = []
    for image in request.files.getlist("images[]"):
        data = image.read()
        data = resize_image(data)
        images.append(data)
    for url in request.form.getlist("images[]"):
        data = curl(url)
        data = resize_image(data)
        images.append(data)
    for data in images:
        box = localize_sess.run(ops, feed_dict={FLAGS.input_node: data})
        app.logger.info("%s %s " % (box, type(box)))
        results.append({'boxpos': box.tolist()})
    response = jsonify(results=results)
    response = make_response(jsonify(results=results))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port) 
