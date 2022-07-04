# -*- encoding:utf-8 -*-
import numpy as np


class LinearRegression:
    def linear_regression(self, w, b, X):
        """线性回归模型"""
        y_hat = np.dot(X, w) + b
        return y_hat

    @staticmethod
    def squared_loss(y_hat, y):
        """均方差损失函数"""
        return 0.5 * np.sum((y_hat - y) ** 2)
