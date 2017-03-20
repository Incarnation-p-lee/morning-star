#!/usr/bin/python

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)

