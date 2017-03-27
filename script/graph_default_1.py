#!/usr/bin/python

import tensorflow as tf

graph = tf.get_default_graph()

input_value = tf.constant(1.0)

operations = graph.get_operations()

print(input_value)
print("--------------------------------")
print(operations)
print("--------------------------------")
print(operations[0].node_def)

with tf.Session() as sess:
    print(sess.run(input_value))

print(input_value)

