#!/usr/bin/python

import tensorflow as tf

maxtrix_1 = tf.constant([[3., 3.]])       # op-1
maxtrix_2 = tf.constant([[2.], [2.]])     # op-2
product = tf.matmul(maxtrix_1, maxtrix_2) # op-3

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        result = sess.run(product)
        print(result)

