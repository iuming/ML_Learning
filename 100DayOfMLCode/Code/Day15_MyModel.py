"""
Program Name: Day15_MyModel
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/21 上午11:04

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/21 上午11:04: Initial Create.

"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.dl = Dense(3,
                        activation = 'softmax',
                        kernel_regularizer=tf.keras.regularizers.l2())
    def call(self, x):
        y = self.dl(x)
        return y

model = IrisModel()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_data,
          y_data,
          batch_size=32,
          epochs=500,
          validation_split=0.2,
          validation_freq=20)

model.summary()