#!/usr/bin/python

import input_data
import tensorflow as tf

sess = tf.Session()

#
# one_hot means only one 1 in label data.
# mnist.train.images are [55000, 28 * 28] tensor
# mnist.train.lables are [55000, 10] (one_hot) tensor
#
mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)

#
# use softmax transfer output to percents
# matmul(hidden, weight) + bias
# 28 * 28 == 784
#
x = tf.placeholder("float", [None, 784], name = 'x')
w = tf.Variable(tf.zeros([784, 10]), name ='w')
b = tf.Variable(tf.zeros([10]), name = 'b')
y = tf.nn.softmax(tf.matmul(x, w) + b, name = 'output')

#
# loss, H(y) = SUM[y * log(y)]
#
y_ = tf.placeholder("float", [None, 10], name = 'y_')
loss = -tf.reduce_sum(y_ * tf.log(y), name = 'loss')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#
# train and test
#
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
    # summary_writer.add_summary(step, i)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy,
               feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

sess.close()

#
# summary data
#
# def var_summary(var):
#    tf.summary.histogram(var.op.name, var)
#    tf.summary.scalar('Max', tf.reduce_max(var))
#    tf.summary.scalar('Min', tf.reduce_min(var))
#
# tf.summary.scalar(loss.op.name, loss)
# var_summary(x)
# var_summary(w)
# var_summary(b)
# var_summary(y)
# summaries = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter('/tmp/tensor_log', sess.graph)

