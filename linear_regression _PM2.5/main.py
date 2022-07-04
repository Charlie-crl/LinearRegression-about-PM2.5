# -*- encoding:utf-8 -*-
import numpy as np
from optimizer import Optimizer
from linear_regression import LinearRegression
from dataset import train_data_process
import pandas as pd
from matplotlib import pyplot as plt

# 繁体字编码使用的是big5
data = pd.read_csv('csv_data/train.csv', encoding='big5')
X, y = train_data_process(data)
# 训练参数
# lr = 0.01
lr = 1
# 多元下
# params = {
#     'w': np.zeros((9, 18)),
#     'b': np.zeros((18, 18))
# }

params = {
    'w': np.zeros(9),
    'b': 0.0
}

epochs = 2000

# 开始训练
h = {}
v = {}
m = {}
beta1 = 0.9
beta2 = 0.999
params_iter = {
    'iter': 0
}
momentum = 0.8
p = 0.99
losses = []
model = LinearRegression()
optimizer = Optimizer()
for i in range(epochs):
    loss = 0
    gradss = {
        'w': np.zeros(9),
        'b': 0.0
    }
    print("epochs:{0}/{1}".format(i, epochs))
    # 一元
    for j in range(len(X)):
        gradss['b'] = gradss['b'] + ((np.dot(X[j, 9], params['w']) + params['b']) - y[j])
        for k in range(9):
            gradss['w'] = gradss['w'] + np.dot(((np.dot(X[j, 9], params['w']) + params['b']) - y[j]), X[j, 9, k])

    # 多元
    # for j in range(len(X)):
    #     gradss['b'] = gradss['b'] + ((np.dot(X[j], params['w']) + params['b']) - y[j])
    #     gradss['w'] = gradss['w'] + np.dot(((np.dot(X[j], params['w']) + params['b']) - y[j]), X[j])

    gradss['b'] = gradss['b'] / len(X)
    gradss['w'] = gradss['w'] / len(X)

    # 转置（多元下）
    # gradss['w'] = np.transpose(gradss['w'])

    # optimizer.momentum(momentum, v, lr, params, gradss)
    # optimizer.adagrad(v, lr, params, gradss)
    # optimizer.rmsprop(p, h, lr, params, gradss)
    optimizer.adam(beta1, beta2, params_iter, m, v, lr, params, gradss)
    # print(params)
    for j in range(len(X)):
        y_hat = model.linear_regression(params['w'], params['b'], X[j, 9])
        loss = loss + model.squared_loss(y_hat, y[j])
    losses.append(loss / len(X))
    print("train_loss:{0}".format(loss / len(X)))

plt.plot(list(range(epochs)), losses)
# plt.title("adagrad lr = " + str(lr))
# plt.title("momentum lr = " + str(lr))
# plt.title("rmsprop lr = " + str(lr))
plt.title("adam lr = " + str(lr))
plt.show()
# np.save("./model/model_w_adagrad" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['w'])
# np.save("./model/model_b_adagrad" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['b'])
# np.save("./model/model_w_momentum" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['w'])
# np.save("./model/model_b_momentum" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['b'])

# np.save("./model/model_w_adam_mul" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['w'])
# np.save("./model/model_b_adam_mul" + "_" + str(epochs) +"epochs" + "_lr=" + str(lr) + ".npy", params['b'])
