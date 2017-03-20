#!/usr/bin/python

import tensorflow as tf

i1 = tf.placeholder(tf.float32)
i2 = tf.placeholder(tf.float32)
output = tf.multiply(i1, i2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={i1:[7.], i2:[2.]}))

