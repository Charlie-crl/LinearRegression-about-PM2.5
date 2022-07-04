# LinearRegression-about-PM2.5
大数据分析作业，关于PM2.5的线性回归预测

###Description

一、

​	本次作业使用丰原站的观测记录，分成 train set 跟 test set，train set 是丰原站每个月的前 20 天所有数据。test set 则是从丰原站剩下的资料中取样出来。
○ train.csv: 每个月前 20 天的完整数据。
○ test.csv : 从剩下的数据当中取样出连续的 10 小时为一笔，前九小时的所有观测数据当作 feature，第十小时的 PM2.5 当作 answer。一共取出 240 笔不重复的 test data，请根据 feature 预测这 240 笔的 PM2.5。
​	Data 含有 18 项观测数据 AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR。

​	预测 240 笔 testing data 中的 PM2.5 值，最好自己不调模型编写代码。

○ Upload format : csv file named submit.csv
○ 第一行必须是 id,value
○ 第二行开始，每行分别为 id 值及预测 PM2.5 数值，以逗号隔开。

二、

​	采用至少两种不同的梯度下降方法, 针对每一种梯度下降方法，调整学习率大小，将loss值和迭代次数的关系用图 展示出来，并尝试解释结果。

三、选取至少两个不同的模型，并使用n折交叉验证的方法对这两个模型进行选择。

​	

### 代码结构

![](.\assert\代码结构.PNG)

我手写了一个线性回归模型类，一个优化器类，包含Momentum，AdaGrad，RMSProp和Adam优化算法，一个对气候数据预处理的dataset接口，返回可以直接训练的X和y。然后就是main主函数，导入数据，训练模型，保存模型，最后就是predict文件进行预测输出submit.csv了。之后与其他模型进行了交叉验证。

### 数据预处理

根据题目意思，即根据前9个小时的数据（作为feature），预测出第十个小时的PM2.5的值。先观察一下train.csv

![](.\assert\train结构.png)

可见，训练数据由黄色部分组成，预测结果为红色部分。根据这样的feature+label组合，一天24小时可以有15个（0~8；9,1~9；10，。。。）这是从横向来看，从纵向来看，有4321行，减去第一行表头，(4321-1)/18 = 240天。所以240天的数据，总共有15*240=3600的训练组合。另外表格中有缺失值，而且前三项属于无关项，进行相应的处理。

分析完后，接下来就是编码。

![](.\assert\dataset1.PNG)

![](.\assert\dataset2.PNG)

### 训练过程

根据线性回归的特点，我选择用一元线性回归来进行训练和预测，所以训练的时候，我只用到了PM2.5那一行数据来进行训练。

#### 参数

lr = 0.00001

epochs = 200

优化器选的momentum，动量因子设成0.9

train loss如图：

![](.\assert\momentum.png)

最后均方差损失收敛在75左右。

#### 调整学习率

尝试调高一点，调到0.001，结果

![](.\assert\bad_momentum.png)

可见train loss一路升高，最后直接起飞了，即“下山”时找了条越走越高的路。如果在x,y轴上画个等高线，momentum算法在y轴方向上的速度不稳定。

#### 更换优化算法

lr = 0.01

epochs = 200

优化器为adagrad

train loss如图：

![](.\assert\adagrad.png)

结果和momentum一样，收敛到75左右

####调整学习率

尝试调低一点，调到0.001，结果

![](.\assert\bad_adagrad.png)

可见还没有收敛。adagrad算法的思想是学习率衰减，到后期更新的幅度越来越小，所以学习率一开始设置比较低的话，就很难收敛了。

### 继续优化

上面的实验我用的是一元线性回归，所以我开始考虑把18个特征都加入进去进行多元线性回归。但实验结果不是很理想，值得我后续继续学习与改进。经过调参与优化算法的选择，得出最佳效果如图：

![](.\assert\adam.png)

train loss降到了71

### 交叉验证

与线性回归相比，在岭回归的正则项系数α 不同的取值下，岭回归的结果如何变化呢？这里我用的是sklearn的cross_val_score函数来做交叉验证。

##### 参数

训练数据集与上面相同

CV = 10（10折交叉验证）

scoring="r2"

α取值范围：np.arange(1, 4000, 10)

结果如图：

![](.\assert\r2.png)

可见岭回归在0~3000左右是由于线性回归的，其中当参数α取值1500左右时效果最好。

####改用scoring="neg_mean_squared_error"试试

如图：

![](.\assert\neg_mean_squared_error2.png)

![](.\assert\neg_mean_squared_error.png)

结果与评估算法为R2时一样。