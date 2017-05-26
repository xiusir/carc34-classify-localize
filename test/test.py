import tensorflow as tf

filenames = [
        '/Users/yangxiu/Workshop/TensorFlow/MyFirstTF/train/o_1b9l01k2537596423402853769643775.jpg',
        '/Users/yangxiu/Workshop/TensorFlow/MyFirstTF/train/o_1b9l01k2537596423402853769643775.jpg',
        ]
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

print key,key.shape
tokens = tf.string_split([key], delimiter='/')
#xx=tokens[tuple(tokens.indices[-1])]
xx=tokens.values[-1]
print xx

yy=tf.image.decode_jpeg(value)
yy.set_shape([256, 256, 3])
print value,yy

#images = tf.image.decode_jpeg(value, channels=3)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    x,y = sess.run([xx,yy])
    print x
    print y
