# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:23:48 2018

@author: xubing
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('F:\\Codes\\Tensorflow\\TensorFlow_Codes\\MNIST_data',
                                  one_hot = True)
#print(mnist.train.images.shape)
x = tf.placeholder(tf.float32,[None,784],name = 'input_x')
W = tf.Variable(tf.zeros([784,10]),name = 'W')
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32,[None,10],name = 'input_y')
#loss = -tf.reduce_sum(tf.log(y_) * y)
x = tf.placeholder()
loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))