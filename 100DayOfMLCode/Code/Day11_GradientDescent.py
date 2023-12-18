"""
Program Name: Day11_GradientDescent
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/17 下午4:20

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/17 下午4:20: Initial Create.

"""

import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print("After %s epoch, w is %f, loss is %f" %(epoch, w.numpy(), loss))
