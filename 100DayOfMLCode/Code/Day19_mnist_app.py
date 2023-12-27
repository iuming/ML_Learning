"""
Program Name: Day19_mnist_app
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 12/27/23 10:43 AM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 12/27/23 10:43 AM: Initial Create.

"""

import tensorflow as tf
from PIL import Image
import numpy as np

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input("Input the number of test Pictures:"))

for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    # img_arr = 255 - img_arr
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0
    img_arr = img_arr / 255.
    x_predict = img_arr[tf.newaxis, ...]
    print('x_predict:', x_predict.shape)
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)