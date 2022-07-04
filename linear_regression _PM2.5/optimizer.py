# -*- encoding:utf-8 -*-

import numpy as np


class Optimizer:
    # 参数更新
    def momentum(self, momentum, v, lr, params, grads):
        """
        考虑连续梯度的上下文关系
        v <- mv - lr * grads
        w <- w + v
        b <- b + v"""
        if v == {}:
            for key, val in params.items():
                v[key] = np.zeros_like(val)

        for key in params.keys():
            v[key] = momentum * v[key] - lr * grads[key]
            params[key] += v[key]

    def adagrad(self, h, lr, params, grads):
        """
        学习率衰减
        h <- h + grads * grads
        w <- w - lr * grads/sqrt(h)
        b <- b - lr * grads/sqrt(h)"""
        if h == {}:
            for key, val in params.items():
                h[key] = np.zeros_like(val)

        for key in params.keys():
            # h[key] += grads[key] * grads[key] # 一元
            # 多元，按元素乘积
            h[key] += np.multiply(grads[key], grads[key])
            params[key] -= lr * grads[key] / (np.sqrt(h[key]) + 1e-7)

    def rmsprop(self, p, h, lr, params, grads):
        """
        在adagrad上引入了p衰减率来避免学习率过早衰减
        h <- p * h + (1-p)* grads * grads
        w <- w - lr * grads/sqrt(h)
        b <- b - lr * grads/sqrt(h)"""
        if h == {}:
            for key, val in params.items():
                h[key] = np.zeros_like(val)

        for key in params.keys():
            h[key] = p * h[key] + (1-p) * np.multiply(grads[key], grads[key])
            params[key] -= lr * grads[key] / (np.sqrt(h[key]) + 1e-7)

    def adam(self, beta1, beta2, iter, m, v, lr, params, grads):
        """
        可以看作动量法和RMSProp算法的结合
        """
        if m == {} and v == {}:
            for key, val in params.items():
                m[key] = np.zeros_like(val)
                v[key] = np.zeros_like(val)
        iter['iter'] = iter['iter'] + 1
        lr_t = lr * np.sqrt(1.0 - beta2**iter['iter'])/(1.0 - beta1**iter['iter'])
        for key in params.keys():
            m[key] += (1 - beta1) * (grads[key] - m[key])
            v[key] += (1 - beta2) * (np.square(grads[key]) - v[key])
            params[key] -= lr_t * m[key] / (np.sqrt(v[key]) + 1e-7)
