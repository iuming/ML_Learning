"""
Program Name: File Converter
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023-10-17

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: This program is using to learn KNN.

Usage: Run the program.

Dependencies:
- Python 3.8 or above
- math library (version 1.3.0 or above)

Modifications:
- 2023-10-17: Initial Create.

"""

# Prepare Training Set
training_set = [
    [100, 200000],
    [150, 250000],
    [120, 220000],
    [180, 280000],
    [200, 300000]
]

import math

# Define Euclidean Diatance
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Predict Price Function
def predict_price(new_house, training_set, k):
    distances = []
    for house in training_set:
        distance = euclidean_distance(new_house, house[:-1])
        distances.append((house, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    prices = [neighbor[0][-1] for neighbor in neighbors]
    return sum(prices) / len(prices)

# Predict new house Price
new_house = [190]  # 预测该房屋的价格
k = 3
predicted_price = predict_price(new_house, training_set, k)
print(predicted_price)
