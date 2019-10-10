#!/usr/bin/env python
# encoding: utf-8
'''
@author: Sunzb
@time: 2019/10/10 9:15
'''
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    a = tf.get_variable(name='a', shape=[1, 3, 3], initializer=tf.constant_initializer(a))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.nn.crelu(a)))
