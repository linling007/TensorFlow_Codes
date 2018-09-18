# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:58:52 2018
tensorflow入门：
用tensorflow做线性回归
@author: xubing
"""
import tensorflow as tf
import numpy as np

#随机造数据
x_data = np.random.rand(100).astype('float32')
y_data = x_data * 0.1 + 0.3

#初始化W和b
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#计算损失，定义目标函数
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train  = optimizer.minimize(loss)

#在会话中运行
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)#初始化以所有变量
    for step in range(201):
        sess.run(train)#优化目标函数
        if step % 20 == 0:
            print(step,sess.run(W),sess.run(b),sess.run(loss))

