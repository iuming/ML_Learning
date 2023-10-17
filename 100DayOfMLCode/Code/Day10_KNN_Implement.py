"""
Program Name: Day10_KNN_Implement
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/17 下午3:49

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above
- numpy library (Version 1.23.5 or above)
- matplotlib library (Version or above)
- pandas library (Version 1.5.3 or above)

Modifications:
- 2023/10/17 下午3:49: Initial Create.

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv ')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))