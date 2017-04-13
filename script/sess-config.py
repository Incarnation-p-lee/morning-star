#!/usr/bin/python

import tensorflow as tf
import input_data

sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))

# only cpu:0 is available

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [3, 2])
    c = tf.matmul(a, b)

print(sess.run(c))

