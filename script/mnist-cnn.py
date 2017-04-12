#!/usr/bin/python

import tensorflow as tf
import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)

#
# functions
#
def weight_var(shape):
    # obtain data from truncated normal distribution
    init = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init)

def bias_var(shape):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                           strides = [1, 2, 2, 1], padding = 'SAME')

#
# variables
#
x        = tf.placeholder("float", [None, 784])
y_       = tf.placeholder("float", [None, 10])
x_image  = tf.reshape(x, [-1, 28, 28, 1])

#
# layers-1 conv and max-pool
# conv2d(input, filter, stride, padding, use_cudnn_on_gpu, name)
#     input  [batch,         in_height,    in_width,    in_channels]
#     filter [filter_height, filter_width, in_channels, out_channels]
#
w_conv_1 = weight_var([5, 5, 1, 32])
b_conv_1 = bias_var([32])
h_conv_1 = tf.nn.relu(conv2d(x_image, w_conv_1) + b_conv_1)
h_pool_1 = max_pool_2x2(h_conv_1)        # 14 * 14

#
# layers-2 conv and max-pool
#
w_conv_2 = weight_var([5, 5, 32, 64])
b_conv_2 = bias_var([64])
h_conv_2 = tf.nn.relu(conv2d(h_pool_1, w_conv_2) + b_conv_2)
h_pool_2 = max_pool_2x2(h_conv_2)        # 7 * 7

w_fc_1 = weight_var([7 * 7 * 64, 1024])  # full connection
b_fc_1 = bias_var([1024])

h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
h_fc_1        = tf.nn.relu(tf.matmul(h_pool_2_flat, w_fc_1) + b_fc_1)

#
# dropout
#
keep_prob   = tf.placeholder("float")
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob) 

#
# output layer
#
w_fc_2 = weight_var([1024, 10])
b_fc_2 = bias_var([10])
y      = tf.nn.softmax(tf.matmul(h_fc_1_drop, w_fc_2) + b_fc_2)

#
# train and evaluate
#
loss       = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct    = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy   = tf.reduce_mean(tf.cast(correct, "float"))

sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})   

print(accuracy.eval(feed_dict = {
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

