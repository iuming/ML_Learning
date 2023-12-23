"""
Program Name: Day16_mnist_train_ex1
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/22 下午12:57

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/22 下午12:57: Initial Create.

"""

import tensorflow as tf
from PIL import Image
import numpy as np
import os

train_path = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000/"
train_text = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000.txt"
x_train_savepath = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_x_train.npy"
y_train_savepath = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_y_train.npy"

test_path = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000/"
test_text = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000.txt"
x_test_savepath = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_x_test.npy"
y_test_savepath = "../../tensorflow2.0/class4/MNIST_FC/mnist_image_label/mnist_y_test_npy"

def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []
    for content in contents:
        value = content.split()
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.0
        x.append(img)
        y.append(value[1])
        print("loading:" + content)

    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int64)
    return x, y

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('------------Loading Data---------------------------')
    x_train_save = np.load(x_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test_save = np.load(y_test_savepath)
    y_train_save = np.load(y_train_savepath)

    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))

else:
    print('------------------Generate Data-----------------------')
    x_train, y_train = generateds(train_path, train_text)
    x_test, y_test = generateds(test_path, test_text)

    print('-------------------Save Data------------------------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test),
          validation_freq=1)

model.summary()