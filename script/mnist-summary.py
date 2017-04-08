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
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
w1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]), name = 'bias')
y1 = tf.matmul(x, w1) + b1

output = tf.nn.softmax(y1)

summary_writer = tf.summary.FileWriter('/tmp/tensor_log', sess.graph)

#
# loss, H(y) = SUM[y * log(y)]
#
loss = tf.reduce_sum(- y_ * tf.log(output), name = 'loss')
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = 'accuracy')
# image_shape = tf.reshape(x, [-1, 28, 28, 1], name = 'image-shape')
# tf.summary.image(image_shape.op.name, image_shape)

#
# Add summary data
#
tf.summary.scalar(loss.op.name, loss)
tf.summary.scalar(accuracy.op.name, accuracy)
# tf.summary.histogram(w1.op.name, w1)
# tf.summary.histogram(b1.op.name, b1)

#
# train and test
#
sess.run(tf.global_variables_initializer())
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
merged = tf.summary.merge_all()

for i in range(1000):
    if i % 5 == 0:
        summary, _ = sess.run([merged, accuracy], feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
        summary_writer.add_summary(summary, i)

    batch_xs, batch_ys = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

print(sess.run(accuracy,
               feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

summary_writer.close()
sess.close()

