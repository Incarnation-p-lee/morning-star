#!/usr/bin/python

import tensorflow as tf

weight = tf.Variable(0.8)
graph = tf.get_default_graph()
sess = tf.Session()

input_value = tf.constant(1.0)
output_value = weight * input_value

#for op in graph.get_operations():
#    print(op.name)
#
#op = graph.get_operations()[-1]
# print(op.name)

init = tf.global_variables_initializer()
sess.run(init)

p = sess.run(output_value)
print(p)

sess.close()

