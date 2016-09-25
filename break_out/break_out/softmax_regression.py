import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf



IMAGE_SIZE = 8
OUTPUT_SIZE = 3

sess = tf.InteractiveSession()
# Create placeholders
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
# Variables three
W = tf.Variable(tf.zeros([IMAGE_SIZE * IMAGE_SIZE, OUTPUT_SIZE]))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))


sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# load data

images = pickle.load(open('data/X.p', 'rb'))
target = pickle.load(open('data/Y.p', 'rb'))

x = images[:1000].flatten()
y_ = target[:1000]

for i in range(1000):
  train_step.run(feed_dict={x: x, y_: y_})