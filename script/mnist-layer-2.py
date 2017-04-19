#!/usr/bin/python

import tensorflow as tf
import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)

#
# variables
#
d         = 784
m         = 784
n         = 10

ref       = tf.placeholder("float", shape = [None, n])

x         = tf.placeholder("float", shape = [None, d])
weight_1  = tf.Variable(tf.truncated_normal([d, m], stddev = 0.04))
bias_1    = tf.constant(0.02, shape = [m])
y_layer_1 = tf.matmul(x, weight_1) + bias_1;


weight_2  = tf.Variable(tf.truncated_normal([m, n], stddev = 0.08))
bias_2    = tf.constant(0.04, shape = [n])
y_layer_2 = tf.matmul(y_layer_1, weight_2) + bias_2;

y         = tf.nn.softmax(y_layer_2)

#
# loss and evaluate
#
loss       = -tf.reduce_sum(ref * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct    = tf.equal(tf.argmax(y, 1), tf.argmax(ref, 1))
accuracy   = tf.reduce_mean(tf.cast(correct, "float"))

#
# train 
#
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict = {x: batch[0], ref: batch[1]})

    if i % 100 == 0:
        print(accuracy.eval(feed_dict = {x: mnist.test.images, ref: mnist.test.labels}))

