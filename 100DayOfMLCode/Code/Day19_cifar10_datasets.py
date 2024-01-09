"""
Program Name: Day19_cifar10_datasets
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 12/27/23 4:37 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 12/27/23 4:37 PM: Initial Create.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.imshow(x_train[0])
plt.show()

print("x_train[0]:\n", x_train[0])

print("y_train[0]:\n", y_train[0])

print("x_test.shape:\n", x_test.shape)