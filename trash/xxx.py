import tensorflow as tf
import numpy as np
import matplotlib;

matplotlib.use('Agg')
import matplotlib.pyplot as plt

filenames = [
        './train/o_1bbaunul1622676372861366335103583.jpg',
        './train/o_1b9l01k2537596423402853769643775.jpg',
        './train/o_1bf4tuqd41589335837444113004721092.jpg',
        './train/o_1b9l01k25395654764865193133765615.jpg',
        './train/o_1bf4tuqd41955710893388202096427215.jpg',
        ]
count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)

reader=tf.WholeFileReader()
key,value=reader.read(filename_queue)
img = tf.image.decode_jpeg(value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    keys = sess.run(key)
    print (keys)
    keys = sess.run(key)
    print (keys)
    ##num_files = sess.run(count_num_files)
    ##for i in range(num_files):
    ##    image=img.eval()
    ##    print(image.shape)
    ##    #Image.fromarray(np.asarray(image)).save('te.jpeg')
