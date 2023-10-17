"""
Program Name: Day8_SVM_Students
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/17 下午2:20

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: 假设我们有一个二分类问题，要根据学生的两门考试成绩（数学和英语）来预测他们是否被大学录取。我们收集了一些训练数据，每个样本包含两门考试成绩和是否被录取的标签。

Usage: Run the program.

Dependencies:
- Python 3.8 or above
- numpy library (version 1.23.5 or above)

Modifications:
- 2023/10/17 下午2:20: Initial Create.

"""

import numpy as np
from sklearn import svm

# 训练数据
X_train = np.array([[90, 85], [85, 80], [30, 40], [20, 10]])
y_train = np.array([1, 1, 0, 0])

# 创建SVM模型
model = svm.SVC(kernel='linear')

# 训练模型
model.fit(X_train, y_train)

# 测试数据
X_test = np.array([[70, 75], [40, 60]])
y_test = np.array([1, 0])

# 预测结果
y_pred = model.predict(X_test)

# 输出预测结果
print("预测结果:", y_pred)


# import numpy as np
#
# # 训练数据
# X_train = np.array([[90, 85], [85, 80], [30, 40], [20, 10]])
# y_train = np.array([1, 1, -1, -1])
#
#
# # 定义SVM模型
# class SVM:
#     def __init__(self, C=1.0):
#         self.C = C
#         self.W = None
#         self.b = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         # 初始化参数
#         alpha = np.zeros(n_samples)
#         bias = 0
#         tol = 1e-4
#         max_iter = 100
#
#         # 计算Gram矩阵
#         K = np.dot(X, X.T)
#
#         # 训练模型
#         for _ in range(max_iter):
#             for i in range(n_samples):
#                 condition = y[i] * (np.sum(alpha * y * K[:, i]) + bias)
#                 if condition < 1:
#                     alpha[i] += 1
#                     bias += y[i]
#
#             # 更新权重
#             self.W = np.dot(X.T, alpha * y)
#             # 更新偏置
#             self.b = np.mean(y - np.dot(X, self.W))
#
#     def predict(self, X):
#         return np.sign(np.dot(X, self.W) + self.b)
#
#
# # 创建SVM模型
# model = SVM()
#
# # 训练模型
# model.fit(X_train, y_train)
#
# # 测试数据
# X_test = np.array([[70, 75], [40, 60]])
#
# # 预测结果
# y_pred = model.predict(X_test)
#
# # 输出预测结果
# print("预测结果:", y_pred)
