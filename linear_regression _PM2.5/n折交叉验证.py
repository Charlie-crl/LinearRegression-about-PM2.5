import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from dataset import train_data_process
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

# 导入数据，分别为输入特征和标签
data = pd.read_csv('csv_data/train.csv', encoding='big5')
X, y = train_data_process(data)
x_list = []
# print(X[0, 9], X[0, 9].shape)
for i in range(len(X)):
    x_list.append(X[i, 9].tolist())
# print(x_list)
y.tolist()
# 验证模型
# scoring：accuracy:精确度  precision_weighted:查准率  recall_weighted:召回率  f1_weighted:f1得分
# cv：几折交叉验证
# n_jobs：同时工作的cpu个数（-1代表全部）

# 交叉验证下，与线性回归相比，岭回归的结果如何变化？
"""如果一个数据集在岭回归中使用各种正则化参数取值下，
模型表现没有明显上升（比如出现持平或者下降），
则说明数据没有多重共线性，顶多是特征之间有一些相关性。
反之，如果一个数据集在岭回归的各种正则化参数取值下，
表现出明显的上升趋势，则说明数据存在多重共线性。"""
alpharange = np.arange(1, 4000, 10)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, x_list, y, cv=10, scoring="r2").mean()
    linears = cross_val_score(linear, x_list, y, cv=10, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("r2")
plt.legend()
plt.show()

