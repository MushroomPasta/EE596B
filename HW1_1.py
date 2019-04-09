# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:03:10 2019

@author: ALLEN
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def forward(x,w1,b1,w2,b2,train=True):
    Z = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
    Z2 = tf.matmul(Z,w2)+b2
    if train:
        return Z2
    else:
        return tf.nn.sigmoid(Z2)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))

x=np.array([[1,0],[0,1],[1,1],[0,0]])
y=np.array([[1],[1],[0],[0]])
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None,2],name='X')
Y = tf.placeholder(tf.float32, [None,1],name='Y')

w1 = init_weights([2,3])
b1 = init_weights([3])
w2 = init_weights([3,1])
b2 = init_weights([1])

y_hat = forward(X,w1,b1,w2,b2)
pred = forward(X,w1,b1,w2,b2,False)

lr = 0.1

epochs = 500


cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_hat,labels=Y))


train = tf.train.GradientDescentOptimizer(lr).minimize(cost)


costs = []


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(epochs):
    sess.run(train,feed_dict={X:x,Y:y})
    c = sess.run(cost,feed_dict={X:x,Y:y})
    costs.append(c)
    if i % 100 == 0:
        print(f"Iteration {i}. Cost:{c}.")
print("train complete")
    
prediction = sess.run(pred,feed_dict={X:x})
print(prediction,np.round(prediction))
W = np.squeeze(sess.run(w1))
b = np.squeeze(sess.run(b1))
print(W)
# =============================================================================
# plt.plot(costs)
# plt.show()
# 
# =============================================================================

# =============================================================================
# plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1]+0.2)])
# plot_y = 1 / W[1] * (-W[0] * plot_x - b)
# 
# plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=100, cmap='viridis')
# plt.plot(plot_x, plot_y, color='k', linewidth=2)
# plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
# plt.show()
# =============================================================================













