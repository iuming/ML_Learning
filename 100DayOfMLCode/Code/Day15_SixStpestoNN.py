"""
Program Name: Day15_SixStpestoNN
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/21 上午9:21

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/21 上午9:21: Initial Create.

"""

# six stpes to Neural Network:
# 1. import
# 2. train, test
# 3. model = tf.keras.models.Sequential([network structure])
# 4. model.compile
# 5. model.fit
# 6. model.summary
#
# models.Sequential examples:
#     straighten layer: tf.keras.layers.Flattern()
#     full connection layer: tf.keras.layers.Dense(Neural numbers, activation = "activate function", kernal_regularizers = tf.keras.reguerizers.1())
#     convolution layer: tf.keras.layers.Conv2D(filter = convolution kernal numbers, kernal_size = convolution kernal size, strides = convolution step length, padding = "valid" or "same")
#     LSTM layer: tf.keras.layers.LSTM()
#
# model.compile(optimizer=optimizer, loss=loss function, metrics=["accuracy"])
# optimizer:
#     "sgd": tf.keras.optimizer.SGD(lr = learning rate, momentum = moment parameter)
#     "adagrad": tf.keras.optimizer.Adagrad(lr = learning rate)
#     "adadelta": tf.keras.optimizer.Adadelta(lr = learning rate)
#     "adam": tf.keras.optimizer.Adam(lr = learning, beta1 = 0.99, beta2 = 0.999)
# loss:
#     "mse": tf.keras.Loss.MeanSquaredError()
#     "sparse_categorial_crossentropy": tf.keras.Loss.SparseCategorialCossentropy(from_logits = False)
# metrics:
#     "accurray": both y_ and y are numbers
#     "categorial_accuray": both y_ and y are independent code
#     "sparse_categorial_accuray": y_ is numbers, y is independent code
#
# model.fit(features of train datasets, labels of train datasets, batch_size= ,epochs= ,vaildation_data=(features of test datasets, labels of test datasets), vaildation_spilt=porprotion ,vaildation_freq= )
#
# model_summary()

import tensorflow as tf
import numpy as np
from sklearn import datasets

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(3,
                                                          activation="softmax",
                                                          kernel_regularizer=tf.keras.regularizers.l2())])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_data,
          y_data,
          batch_size=32,
          epochs=500,
          validation_split=0.2,
          validation_freq=20)

model.summary()