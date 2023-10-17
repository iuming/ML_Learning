"""
Program Name: Day9_SVM_Advanced
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/17 下午2:29

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/10/17 下午2:29: Initial Create.

"""


# 支持向量机（Support Vector Machine，SVM）是一种常用的机器学习算法，主要用于分类和回归问题。它的基本思想是通过找到一个最优的超平面来将不同类别的样本点分开。
#
# SVM的工作原理如下：
# 1. 数据预处理：将训练数据进行标准化处理，使得不同特征具有相同的尺度。
# 2. 寻找最优超平面：SVM的目标是找到一个能够最大化分类间隔的超平面。分类间隔是指超平面到最近的样本点的距离。通常使用支持向量来表示最近的样本点。
# 3. 转化为优化问题：将寻找最优超平面的问题转化为一个凸优化问题，即求解一个二次规划问题。
# 4. 核函数的使用：对于线性不可分的问题，可以使用核函数将数据映射到高维空间，使得数据在高维空间中线性可分。
# 5. 模型训练和预测：通过求解优化问题得到最优超平面的参数，然后使用这些参数进行分类预测。
#
# K最近邻算法（K-Nearest Neighbors，KNN）是一种基于实例的学习算法，主要用于分类和回归问题。它的基本思想是通过测量不同样本之间的距离来进行分类。
#
# KNN的实现步骤如下：
# 1. 数据预处理：将训练数据进行标准化处理，使得不同特征具有相同的尺度。
# 2. 计算距离：根据选择的距离度量方法（如欧氏距离、曼哈顿距离等），计算测试样本与训练样本之间的距离。
# 3. 选择K值：确定K值，即选择最近的K个训练样本作为预测结果的依据。
# 4. 多数表决：对于分类问题，根据K个最近邻样本的标签，采用多数表决的方式确定测试样本的分类结果。对于回归问题，可以采用K个最近邻样本的平均值作为预测结果。
# 5. 模型训练和预测：将上述步骤应用于所有测试样本，得到最终的分类或回归结果。
#
# 需要注意的是，SVM和KNN算法都有一些参数需要调优，如SVM中的惩罚参数C和核函数选择，KNN中的K值选择等。调优这些参数可以进一步提高算法的性能。

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 创建示例数据集
np.random.seed(42)
X = np.random.rand(1000, 2)
Y = np.random.randint(0, 2, 1000)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# 创建SVM分类器
svm_classifier = SVC(kernel='linear')

# 拟合训练数据
svm_classifier.fit(X_train, Y_train)

# 预测测试数据
svm_predictions = svm_classifier.predict(X_test)

# 计算准确率
svm_accuracy = accuracy_score(Y_test, svm_predictions)
print("SVM准确率：", svm_accuracy)

# 创建KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# 拟合训练数据
knn_classifier.fit(X_train, Y_train)

# 预测测试数据
knn_predictions = knn_classifier.predict(X_test)

# 计算准确率
knn_accuracy = accuracy_score(Y_test, knn_predictions)
print("KNN准确率：", knn_accuracy)

# 要提高分类模型的准确率，可以尝试以下几种方法：
#
# 1. 数据预处理：对数据进行预处理是提高准确率的重要步骤。包括特征缩放、特征选择、处理缺失值、处理异常值等。适当的数据预处理可以提高模型的稳定性和性能。
#
# 2. 特征工程：通过添加新的特征或对现有特征进行变换，可以提供更多的信息给模型。例如，可以使用多项式特征、交叉特征、特征组合等方法来提取更丰富的特征表示。
#
# 3. 参数调优：选择合适的模型参数可以显著影响模型的性能。可以使用交叉验证等技术来搜索最佳参数组合。例如，对于SVM模型，可以调整核函数、正则化参数等；对于KNN模型，可以调整最近邻数目等。
#
# 4. 模型集成：使用模型集成方法，如投票（Voting）、堆叠（Stacking）、提升（Boosting）等，可以将多个模型的预测结果结合起来，从而提高整体的准确率。
#
# 5. 处理不平衡数据：如果数据集中的类别分布不平衡，即某些类别的样本数量较少，可以采用过采样（Oversampling）或欠采样（Undersampling）等方法来平衡类别分布，从而提高模型对少数类别的预测能力。
#
# 6. 改变模型算法：尝试不同的算法或模型结构，以找到最适合数据集的模型。不同的算法可能对不同类型的数据和问题具有更好的适应性。
#
# 7. 增加训练样本数量：增加更多的训练样本可以提供更多的信息给模型，从而提高模型的准确率。
#
# 需要根据具体问题和数据集的特点来选择适合的方法。同时，还应该注意避免过拟合，确保模型的泛化能力。