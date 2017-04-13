#!/usr/bin/python

import tensorflow as tf

def my_image_filter(image):
    conv1_w    = tf.Variable(tf.random_normal([5, 5, 32, 32])
    conv1_bias = tf.Variable(tf.zeros([32]))
    conv1      = tf.nn.conv2d(image, conv1_w,
                              strides = [1, 1, 1, 1], padding = 'SAME')
    relu1      = tf.nn.relu(conv1 + conv1_bias)

    conv2_w    = tf.Variable(tf.random_normal([5, 5, 32, 32])
    conv2_bias = tf.Variable(tf.zeros([32]))
    conv2      = tf.nn.conv2d(image, conv2_w,
                              strides = [1, 1, 1, 1], padding = 'SAME')
    relu2      = tf.nn.relu(conv2 + conv2_bias)

    return relu2

#
# if we call my_image_filter twice like
#
my_image_filter(image_1)
my_image_filter(image_2)

#
# It has double count of variable conv1_w conv1_bias conv1 relu1
# and conv2_w conv2_bias conv2 relu2
# Use tf.get_variable(name, shape, initializer)
#     tf.variable_scope(name)
# initializer: tf.constant_initializer(value)
#              tf.random_uniform_initializer(a, b)
#              tf.random_normal_initializer(mean, stddev)
#
def conv_relu(image, kernel_shape, bias_shape):
    w    = tf.get_variable('w', kernel_shape,
                           initializer = tf.random_normal_initializer())
    b    = tf.get_variable('b', bias_shape,
                           initializer = tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(image, w, strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + b)

def my_image_filter(image):
    with tf.variable_scope("conv_1"):
        # create variable conv1/w conv1/b
        relu_1 = conv_relu(image, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv_2"):
        # create variable conv_2/w conv_2/b
        return conv_relu(relu_1, [5, 5, 32, 32], [32])

my_image_filter(image_1)
my_image_filter(image_2)
#
# and will raise ValueError ... conv_1/w already exists.
#

with tf.variable_scope("image_filter") as scope:
    result_1 = my_image_filter(image_1)
    scope.reuse_variables()
    result_2 = my_image_filter(image_2)

#
# other formats
#
with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1])
with tf.variable_scope('foo', reuse = True): # will serach v and return
    w = tf.get_variable('v', [1])
assert(v == w)

with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1])
    tf.get_variable_scope().reuse_variables()
    w = tf.get_variable('v', [1])
assert(v == w)

#
# get variable scope
#
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v', [1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable('w', [1])
with tf.variable_scope(foo_scope, reuse = True):
    v1 = tf.get_variable('v', [1])
    w1 = tf.get_variable('w', [1])
assert (v == v1 &&  w == w1)


with tf.variable_scope('foo') as foo_scope:
with tf.variable_scope('bar'):
    with tf.variable_scope('dummy') as d_scope:
        assert(d_scope.name == 'bar/dummy')
        with tf.variable_scope(foo_scope) as foo_2_scope:
            assert(foo_2_scope.name = 'foo')


with tf.variable_scope('foo', initializer = tf.constant_initializer(0.4)):
    v = tf.get_variable('v', [1])
    assert(v.eval() == 0.4)
    w = tf.get_variable('w', [1], initializer = tf.constant_initializer(0.9)):
    assert(w.eval() == 0.9)
    with tf.variable_scope('bar'):
        v = tf.get_variable('v', [1])
        assert(v.eval() == 0.4)
    with tf.variable_scope('dummy', initializer = tf.constant_initializer(0.1)):
        v = tf.get_variable('v', [1])
        assert(v.eval() == 0.1)


with tf.variable_scope('foo'):
    x = 1.0 + tf.get_variable('v', [1])
assert(x.op.name == 'foo/add')

with tf.variable_scope('foo'):
    with tf.name_scope('dummy'): # DO not effect on variables
        v = tf.get_variable('v', [1])
        x = 1.0 + v
assert(v.name == 'foo/v:0')
assert(x.name == 'foo/dummy/add')

