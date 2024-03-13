"""
Program Name: Day24_RNN_GRU
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 1/24/24 7:23 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 1/24/24 7:23 PM: Initial Create.

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

maotai = pd.read_csv('../datasets/SH600519.csv')

training_set = maotai.iloc[0:2426-300, 2: 3].values
test_set = maotai.iloc[2426-300:, 2: 3]

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.fit_transform(test_set)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

for i in range(60, len(test_set)):
    x_test.append(test_set[i-60:i, 0])
    y_test.append(test_set[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.models.Sequential([
    GRU(80, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

checkpoint_save_path = './checkpoint/rnn_GRU_stock.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True)

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=50,
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

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(test_set[60:])

plt.plot(real_stock_price, color='red', label='Maotai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Maotai Stock Price')
plt.title('Maotai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Maotai Stock Price')
plt.legend()
plt.show()

mse = mean_squared_error(predicted_stock_price, real_stock_price)
rmse = math.sqrt(mse)
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('mean_squared_error: %.6f' %mse)
print('sqreat mean_squared_error: %.6f' %rmse)
print('mean_absolute_error:%.6f' %mae)