"""
Program Name: Day15_fashion_class
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/21 下午3:51

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/21 下午3:51: Initial Create.

"""

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class FashionClass(Model):
    def __init__(self):
        super(FashionClass, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        y = self.dense2(x)
        return y

model = FashionClass()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=20,
          validation_data=(x_test, y_test),
          validation_freq=1)

model.summary()