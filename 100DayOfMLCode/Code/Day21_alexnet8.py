"""
Program Name: Day21_alexnet8
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 1/15/24 9:30 AM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 1/15/24 9:30 AM: Initial Create.

"""

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.

class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8,self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')
        self.b1 = BatchNormalization()
        self.a1 = Activation(activation='relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)
        # self.d1 = Dropout(0)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)
        # self.d2 = Dropout(0)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same')
        # self.b3 = BatchNormalization(False)
        self.a3 = Activation('relu')
        # self.p3 = MaxPool2D(pool_size=(1, 1), strides=1)
        # self.d3 = Dropout(0)

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same')
        # self.b4 = BatchNormalization(False)
        self.a4 = Activation('relu')
        # self.p4 = MaxPool2D(pool_size=(1, 1), strides=1)
        # self.d4 = Dropout(0)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        # self.b5 = BatchNormalization(False)
        self.a5 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(3, 3), strides=2)
        # self.d5 = Dropout(0)

        self.f = Flatten()
        self.dense1 = Dense(2048, activation='relu')
        self.drop1 = Dropout(0.5)
        self.dense2 = Dense(2048, activation='relu')
        self.drop2 = Dropout(0.5)
        self.dense3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.a4(x)

        x = self.c5(x)
        x = self.a5(x)
        x = self.p5(x)

        x = self.f(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        y = self.dense3(x)
        return y

model = AlexNet8()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/AlexNet8.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------------load the model--------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True)
history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()