#!/usr/bin/python

import tensorflow as tf

sess = tf.Session()

hello = tf.constant('Hello, Tensorflow');
a = tf.constant(30)
b = tf.constant(23)

print(sess.run(hello))
print(sess.run(a + b))

