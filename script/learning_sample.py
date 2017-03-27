#!/usr/bin/python

import tensorflow as tf

sess = tf.Session()

x = tf.constant(1.0, name = 'input')
w = tf.Variable(0.8, name = 'weight')
y = tf.multiply(x, w, name = 'output');
y_ = tf.constant(0.0, name = 'expected_value')
loss = tf.pow(y - y_, 2, name = 'loss')

train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/tensor_log', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)
    # print(sess.run(y))

sess.close()

