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
dataset.head(5)