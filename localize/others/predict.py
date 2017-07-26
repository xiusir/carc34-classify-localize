import numpy as np
import tensorflow as tf

labelsFullPath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/enlarge/new/label_strings.txt'

imagePath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34/image/24/cn/o_1bbig1v504226328749310003144644059.jpg'
imagePath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/tmp/carc34_200/image/21/cn/o_1b37av55l2761979814439221740660.jpg'

modelFullPath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/release/inference.pb'
modelFullPath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/release/bazel.pb'
modelFullPath = '/home/xiusir/WorkShop/TensorFlow/MySecondModel/model/frozen_custom.pb'
#output_node = 'softmax_linear/softmax_linear:0'
output_node = 'output:0'
input_node = 'input_image:0'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


IMAGE_CHANNEL=3
IMAGE_SIZE=256

def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    image = tf.image.decode_jpeg(image_data, channels=IMAGE_CHANNEL)
    #image0 = tf.cast(image, tf.float32)
    #image0 = tf.reshape(image0, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    #image1 = tf.image.per_image_standardization(image0)
    #images = tf.reshape(image1, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    images = image

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        images = sess.run(images)

        output_tensor = sess.graph.get_tensor_by_name(output_node)
        predictions = sess.run(output_tensor, {input_node: images})
        predictions = np.squeeze(predictions[0])

        top_k = predictions.argsort()[-10:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (str(human_string), score))

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':
    run_inference_on_image()

#filename: o_1bbig1v504226328749310003144644059.jpg | probabilities: 24,车身外观/车身外观-外观-右后,15.649 29,车身外观/车身外观-外观-正后,9.028 14,中控内饰/中控内饰-车门,6.776
#filename: o_1bd6fal25519371325941296146542890.jpg | probabilities: 25,车身外观/车身外观-外观-左前,33.088 24,车身外观/车身外观-外观-右后,14.679 27,车身外观/车身外观-外观-正侧,10.329


#车身外观/车身外观-外观-右后 (score = 19.18050)
#车身外观/车身外观-外观-正后 (score = 10.31021)
#中控内饰/中控内饰-车门 (score = 8.63865)
#车身外观/车身外观-外观-左后 (score = 7.93635)
#车身外观/车身外观-外观-左前 (score = 5.82935)
#车身外观/车身外观-外观-正侧 (score = 3.42092)
#车身外观/车身外观-外观-正前 (score = 3.36477)
#中控内饰/中控内饰-内饰-前排 (score = 2.98654)
#车身外观/车身外观-右后尾灯 (score = 2.17646)
#车身外观/车身外观-左后尾灯 (score = 1.76689)


#车身外观/车身外观-外观-右后 (score = 15.64872)
#车身外观/车身外观-外观-正后 (score = 9.02837)
#中控内饰/中控内饰-车门 (score = 6.77566)
#车身外观/车身外观-外观-左后 (score = 6.67561)
#车身外观/车身外观-外观-左前 (score = 4.35854)
#车身外观/车身外观-外观-正前 (score = 3.95414)
#中控内饰/中控内饰-油门踏板 (score = 2.43512)
#车身外观/车身外观-左后尾灯 (score = 2.10728)
#车身外观/车身外观-外观-正侧 (score = 1.91927)
#车身外观/车身外观-右后尾灯 (score = 1.90394)
