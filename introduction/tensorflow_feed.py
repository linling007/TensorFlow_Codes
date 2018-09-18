# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:13:38 2018

@author: xubing
"""

import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)
#tf.mul  tf.sub   tf.neg 已经废弃 
#分别可用tf.multiply  tf.subtract  tf.negative替代.

with tf.Session() as sess:
    print(sess.run([output],feed_dict = {input1:[7.],input2:[8.]}))
    #注意是sess.run([],feed_dict = {})
    