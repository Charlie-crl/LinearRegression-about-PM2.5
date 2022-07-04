# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np


# 训练数据预处理
def train_data_process(data):
    # x_list,y_list分别为feature和label
    x_list, y_list = [], []
    # 没有数据的地方补0
    data = data.replace(['NR'], [0.0])
    # 删除前三列无关项
    data.drop(['date', 'station', 'testmaterial'], axis=1, inplace=True)
    # 转换为numpy数组
    array = np.array(data).astype(float)
    # print(array)
    # 数据集拆分为训练数据和标签
    for i in range(0, len(data.values), 18):
        for j in range(24 - 9):
            # 18个有关特征，间隔九小时为一次feature
            x_t = array[i:i + 18, j:j + 9]
            # print(type(x_t))
            # print(x_t)

            # 第10个指标为pm2.5，对应第十个小时为label
            y_t = array[i + 9, j + 9]
            # print(y_t)
            # 每次都插入到训练数据和标签
            x_list.append(x_t)
            y_list.append(y_t)

    x = np.array(x_list)
    y = np.array(y_list)
    # 转换为numpy数组返回
    return x, y


# 单元测试
if __name__ == '__main__':
    data = pd.read_csv('csv_data/train.csv', encoding='big5')
    x, y = train_data_process(data)

