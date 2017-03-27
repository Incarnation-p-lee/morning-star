#!/usr/bin/python

import tensorflow as tf

sess = tf.Session()

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.add(x, w, name='tmp')
mul = tf.multiply(x, y, name='output');
init = tf.global_variables_initializer()


sess.run(init)
output = sess.run(mul)

summary_writer = tf.summary.FileWriter('/tmp/tensor_log', sess.graph)

#sess.close()

