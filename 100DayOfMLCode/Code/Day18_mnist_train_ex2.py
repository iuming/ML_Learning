"""
Program Name: Day18_mnist_train_ex2
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 12/26/23 11:09 AM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 12/26/23 11:09 AM: Initial Create.

"""

import tensorflow as tf
import numpy as up
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale = 1. / 255.,
    rotation_range = 5,
    width_shift_range = .015,
    height_shift_range = .015,
    horizontal_flip = False,
    zoom_range = 0.5
)

image_gen_train.fit(x_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(False),
              metrics=['sparse_categorical_accuracy'])

model.fit(
    image_gen_train.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test),
    validation_freq=1
)

model.summary()