"""
Program Name: Day13_PreparedFunction
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/19 下午3:14

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/19 下午3:14: Initial Create.

"""

import tensorflow as tf

a = tf.constant([1,2,3,4,5,6])
b = tf.constant([6,5,4,3,2,1])

c = tf.where(tf.greater(a,b) ,a, b)
print(c)

import numpy as np

rdn = np.random.RandomState(seed=1)
a = rdn.rand()
b = rdn.rand(2,3)
print("a:", a)
print("b:", b)

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack((a, b))
print("c:\n", c)

x, y = np.mgrid[1:5:1, 2:4:0.5]
gird = np.c_[x.ravel(), y.ravel()]
print("x:", x)
print("y:", y)
print("gird:", gird)

epoch = 10
w = tf.Variable(tf.constant(5, dtype=tf.float64))
lr = 0.2
loss_all = 0

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grad = tape.gradient(loss, w)
    w.assign_sub(lr * grad)
    print("After %s epoch, w is %s, loss is %f"% (epoch, w.numpy(), loss))

print("-------------------------------------")

LR_BASE = 0.2
LR_DECAY = 0.99
LR_STEP = 1
epoch = 10

for epoch in range(epoch):
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grad = tape.gradient(loss, w)
    w.assign_sub(lr * grad)
    print("After %s epoch, w is %s, loss is %f"% (epoch, w.numpy(), loss))


loss_ce1 = tf.losses.categorical_crossentropy([1., 0.], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1., 0.], [0.8, 0.2])
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)

y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
loss_y = tf.nn.softmax_cross_entropy_with_logits(y_, y)
y_pro = tf.nn.softmax(y)
loss_ypro = tf.losses.categorical_crossentropy(y_, y_pro)
print("loss_y:\n", loss_y)
print("loss_ypro:\n", loss_ypro)
