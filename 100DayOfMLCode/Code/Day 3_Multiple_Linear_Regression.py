import pandas as pd
import numpy as np


dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values
Z = dataset.iloc[ : ,  0 ].values

print("X:")
print(X[:10])
print("Y:")
print(Y)
print("Z:")
print(Z)
# dataset.head(5)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()