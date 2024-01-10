"""
Program Name: Day20_cifar10_lenet5
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 1/9/24 1:01 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 1/9/24 1:01 PM: Initial Create.

"""
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6,
                        kernel_size=(5, 5),
                        activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2),
                            strides=2)
        self.c2 = Conv2D(filters=16,
                        kernel_size=(5, 5),
                        activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2),
                            strides=2)
        self.flatten = Flatten()
        self.d1 = Dense(120, activation='sigmoid')
        self.d2 = Dense(84, activation='sigmoid')
        self.d3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y

model = LeNet5()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/LeNet5.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------------load the model--------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                 save_weights_only=True,
                                 save_best_only=True)

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

print('---------------plot the results----------------')
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_acc, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()