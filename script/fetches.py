#!/usr/bin/python

import tensorflow as tf

in1 = tf.constant(3.0)
in2 = tf.constant(2.0)
in3 = tf.constant(5.0)

intermed = tf.add(in2, in3)
mul = tf.multiply(in1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

