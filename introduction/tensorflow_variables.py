# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:39:49 2018

@author: xubing
"""

import tensorflow as tf
state = tf.Variable(0,name = 'counter')
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)
#tf.assign(A, new_number): 
#这个函数的功能主要是把A的值变为new_number

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))