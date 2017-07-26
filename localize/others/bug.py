# from http://stackoverflow.com/questions/41920371/tensorflow-multi-threaded-queuerunner?noredirect=1#comment71036438_41920371
import tensorflow as tf
import numpy as np

a = tf.train.string_input_producer([str(i) for i in range(1,10)], shuffle=False).dequeue()
b = tf.train.string_input_producer([str(i) for i in range(1,10)], shuffle=False).dequeue()
op1 = tf.identity(a)
op2 = tf.identity(op1)
c1, c2 = tf.train.batch([op2,b], num_threads=5, batch_size=2)

with tf.Session() as sess, tf.device('/gpu:0'):
    sess.run([tf.initialize_all_variables()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(100):
        d1, d2 = sess.run([c1,c2])
        if d1[0] != d2[0] or d1[1] != d2[1]:
            print (d1, d2)

    coord.request_stop()
    coord.join(threads)

