# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import csv

# 数据导入
test_data = pd.read_csv('csv_data/test.csv', encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
array = np.array(test_data).astype(float)
x_list = []
for i in range(0, len(test_data.values), 18):
    x_test = array[i:i + 18, :]
    x_list.append(x_test)

# 模型导入
w = np.load('model/model_w_adagrad_200epochs_lr=0.01.npy')
b = np.load('model/model_b_adagrad_200epochs_lr=0.01.npy')

with open('csv_data/submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)

    for i in range(240):
        row = ['id_' + str(i), np.dot(np.array(x_list[i][9]), w) + b]
        csv_writer.writerow(row)

