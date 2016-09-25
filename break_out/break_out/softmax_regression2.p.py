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

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# load data

images = pickle.load(open('data/X.p', 'rb'))
target = pickle.load(open('data/Y.p', 'rb'))

x_data = images[:1000].reshape(1000,IMAGE_SIZE*IMAGE_SIZE)/256
y__data = target[:1000]

x_data_out = images[1000:2000].reshape(1000,IMAGE_SIZE*IMAGE_SIZE)
y__data_out = target[1000:2000]

print(x_data.shape)
print('p')
print(y__data.shape)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(50000):
    train_step.run(feed_dict={x: x_data, y_: y__data})
    if i % 1000 == 0:
        print(accuracy.eval(feed_dict={x: x_data, y_: y__data}))