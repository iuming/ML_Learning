"""
Program Name: Day17_show_augmented_images
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/24 下午4:56

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/24 下午4:56: Initial Create.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale = 1. /255.,
    rotation_range = 45,
    width_shift_range = .15,
    height_shift_range = .15,
    horizontal_flip = False,
    zoom_range = 0.5
)

image_gen_train.fit(x_train)
print("x_train.shape:", x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("x_train_subset1.shape:", x_train_subset1.shape)
print("x_train.shape:", x_train.shape)
x_train_subset2 = x_train[:12]
print("x_train_subset2.shape:", x_train_subset2.shape)

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i+1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of original training images', fontsize=20)
plt.show()

fig = plt.figure(figsize=(20, 2))
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented images', fontsize=20)
    plt.show()
    break;