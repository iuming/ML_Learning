"""
Program Name: Day12_Iris
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/18 上午11:38

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/18 上午11:38: Initial Create.

"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

x_data = load_iris().data
y_data = load_iris().target

import numpy as np

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

import tensorflow as tf

tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=1, dtype=tf.float64))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1, dtype=tf.float64))

lr = 0.1
train_loss_results = []
test_acc = []
epoch = 250
loss_all = 0

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x_train = tf.cast(x_train, tf.float64)
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y = tf.cast(y, tf.float64)
            y_ = tf.one_hot(y_train, depth=3)
            y_ = tf.cast(y_, tf.float64)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------")

import matplotlib.pyplot as plt

plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()