#!/usr/bin/python

import tensorflow as tf
import input_data
import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)

#
# variables
#
d      = 784
m      = 10
x      = tf.placeholder("float", shape = [None, d], name = 'input')
ref    = tf.placeholder("float", shape = [None, m], name = 'label')
weight = tf.Variable(tf.zeros([d, m]),              name = 'weight')
bias   = tf.Variable(tf.zeros([m]),                 name = 'bias') 
y      = tf.nn.softmax(tf.matmul(x, weight) + bias)

#
# loss and evaluate
#
loss       = -tf.reduce_sum(ref * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
correct    = tf.equal(tf.argmax(y, 1), tf.argmax(ref, 1))
accuracy   = tf.reduce_mean(tf.cast(correct, "float"))

#
# train 
#
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict = {x: batch[0], ref: batch[1]})

print(accuracy.eval(feed_dict = {x: mnist.test.images, ref: mnist.test.labels}))

