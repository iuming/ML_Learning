"""
Program Name: Day18_mnist_train_ex3
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 12/26/23 5:07 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 12/26/23 5:07 PM: Initial Create.

"""

import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(False),
    metrics=['sparse_categorical_accuracy']
)

checkpoint_save_path = './checkpoint/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_best_only=True,
    save_weights_only=True
)

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()