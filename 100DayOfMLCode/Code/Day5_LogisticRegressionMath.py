# 逻辑回归背后的数学是基于线性回归模型和逻辑函数的组合。
#
# 首先，逻辑回归是一种二分类模型，用于预测一个样本属于某一类别的概率。它基于线性回归模型，通过寻找最佳的线性拟合来建立一个决策边界。
#
# 假设有一个二分类问题，其中输入特征为 $x$，输出为 $y$。逻辑回归模型通过计算输入特征与权重的线性组合，再通过一个逻辑函数将结果映射到一个概率值，表示样本属于某一类别的概率。逻辑函数通常使用 sigmoid 函数，其数学表达式为：
#
# $$f(x) = \frac{1}{1 + e^{-z}}$$
#
# 其中，$z$ 表示输入特征与权重的线性组合，可以表示为：
#
# $$z = w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n$$
#
# 在逻辑回归中，我们的目标是找到一组最佳的权重参数 $w$，使得模型的预测结果与实际输出 $y$ 尽可能接近。为了达到这个目标，通常使用最大似然估计或最小化损失函数的方法来优化模型参数。
#
# 常用的损失函数是交叉熵损失函数，它可以表示为：
#
# $$J(w) = -\frac{1}{m} \sum [y\log(f(x)) + (1-y)\log(1-f(x))]$$
#
# 其中，$m$ 表示样本数量，$y$ 表示实际输出，$f(x)$ 表示模型的预测结果。
#
# 通过最小化损失函数，可以使用梯度下降等优化算法来求解最佳的权重参数 $w$。最终得到的 $w$ 可以用于预测新样本的类别概率。
#
# 总结起来，逻辑回归背后的数学包括线性回归模型和逻辑函数的组合，以及使用最大似然估计或最小化损失函数的方法来优化模型参数。

import numpy as np
from sklearn.linear_model import LogisticRegression

# 创建样本数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 0, 1, 1])

# 创建逻辑回归模型
model = LogisticRegression()

# 拟合模型
model.fit(X, y)

# 预测新样本
new_samples = np.array([[6, 7], [1, 1]])
predictions = model.predict(new_samples)

print("预测结果:", predictions)