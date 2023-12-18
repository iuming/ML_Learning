"""
Program Name: Day11_Tensor
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/17 下午4:52

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/17 下午4:52: Initial Create.

"""

import tensorflow as tf
import numpy as np

n = np.arange(0, 5)
nconv = tf.convert_to_tensor(n, dtype=tf.int64)

print(n)
print(nconv.numpy())

a = tf.zeros([2,3])
print(a.numpy())

b = tf.ones(4)
print(b.numpy())

c = tf.fill([2,2], 9)
print(c.numpy())

d = tf.random.normal([2,2], mean=0, stddev=1)
print(d)

e = tf.random.truncated_normal([2,2], mean=0, stddev=1)
print(e)

f = tf.random.uniform([2,2], minval=-1, maxval=1)
print(f)


x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print(x1)

x2 = tf.cast(x1, dtype=tf.int64)
print(x2)
print(tf.reduce_min(x2), tf.reduce_max(x2))

x = tf.constant([[1,2,3],[4,5,6]])
print(x)


print(tf.reduce_mean(x))
print(tf.reduce_sum(x, axis=1))

w = tf.Variable(tf.random.normal([2,2], mean=0, stddev=1))

a1 = tf.ones([1,3], dtype=tf.float32)
a2 = tf.fill([1,3], 2.)

print(tf.add(a1,a2))
print(tf.subtract(a1,a2))
print(tf.multiply(a1,a2))
print(tf.divide(a1,a2))


print(tf.square(x))
print(tf.pow(a2, 3))
print(tf.sqrt(a2))

b1 = tf.random.normal([4,5], mean=10, stddev=5, dtype=tf.float32)
b2 = tf.random.uniform([5,4], minval=-5, maxval=5, dtype=tf.float32)

print(tf.matmul(b1,b2))

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)
print(grad)


seq = ["zero", "one", "two", "three"]

for i, element in enumerate(seq):
    print(i, element)

classes = 4
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print(output)

y = tf.constant([1.1, 2.4, -0.4])
y_pro = tf.nn.softmax(y)
print(y_pro)

w = tf.Variable(4)
w.assign_sub(1)
print(w)

test = np.array([[1,2,3],[2,2,4],[5,4,3],[8,7,2]])
print(test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))