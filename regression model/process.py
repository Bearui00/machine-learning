from random import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


def level_1_a(filename):
    #最小二乘
    data_regression=pd.read_csv(filename,encoding='utf-8')
    data_x=np.array(data_regression.iloc[:, 1]).T
    data_y=np.array(data_regression.iloc[:, 2]).T
    variance=np.var(data_x,ddof=1) #求x方差
    covariance=np.cov(data_x,data_y)[0][1] #求xy协方差
    w=covariance/variance
    b=np.mean(data_y)-w*np.mean(data_x)
    #回归曲线
    print("w = %f\nb = %f" % (w, b))
    predict_y=w*data_x+b
    plt.plot(data_x, data_y, 'k.')  # 样本点
    plt.plot(data_x, predict_y, 'b-')  # 手动求出的线性回归模型
    #训练误差
    MeanSquareEstimate=np.linalg.norm(predict_y - data_y) / data_x.shape[0]
    print("训练误差%f" % MeanSquareEstimate)
    #测试误差
    noise = np.random.normal(loc=0, scale=np.sqrt(1), size=5)
    test_x = np.array(range(-2, 3, 1))
    test_y = b + w * test_x + noise
    predict_y_test=w*test_x+b
    MeanSquareEstimate_test=np.linalg.norm(predict_y_test - test_y) / test_y.shape[0]
    print("测试误差%f" % MeanSquareEstimate_test)
    plt.show()
    return w,b

def Gradient_descent(x, y, w, learn_rate):
    # x:特征向量，1~D列为读取数据，0列全1，N行
    # y：值，1列N行
    # w:规划得到参数，维度同x列维度，(D+1)*1
    # MSE:误差
    MSE = []
    MSE.append(100000)
    MSE.append(10000)
    time_count = 0
    while np.abs(MSE[-1] - MSE[-2]) > 0.000001:
        w_temp = np.zeros((w.shape[0], 1))
        for j in range(w.shape[0]):  # 对所有的参数赋值
            w_temp[j] = w[j] + learn_rate * np.dot((y - np.dot(x, w)).T, x[:, j]) / x.shape[0]
        w = w_temp  # 更新参数
        MSE.append(np.linalg.norm(np.dot(x, w) - y) ** 2 / x.shape[0])
        # print(err[-1])
        time_count += 1
        if np.abs(MSE[-1] - MSE[-2]) > 10 and time_count > 3:
            print('learn rate too big!')
            break
    plt.plot(MSE[2:])
    plt.ylabel('MSE')
    plt.xlabel('times')
    plt.show()
    return time_count, w, MSE

def Stochastic_gradient_descent(x, y, w, learn_rate):
    MSE = []
    MSE.append(100000)
    MSE.append(10000)
    y = y.reshape(y.shape[0], 1)
    time_count = 0
    while np.abs(MSE[-1] - MSE[-2]) > 0.00001:
        err_temp = 0
        for i in random.sample(range(x.shape[0]), x.shape[0]):  # 对排序后样本
            w_temp = np.zeros((w.shape[0], 1))
            for j in range(w.shape[0]):  # 对所有参数求梯度
                w_temp[j] = w[j] + learn_rate * (y[i] - np.dot(x[i, :], w)) * x[i, j]  # 求新的参数
            w = w_temp  # 更新参数
            err_temp += (np.dot(x[i, :], w) - y[i]) ** 2
        MSE.append(err_temp / x.shape[0])  # 计算MSE
        # print(err[-1])
        time_count += 1
        if np.abs(MSE[-1] - MSE[-2]) > 10 and time_count > 100000:
            print('learn rate too big!')
            break
    plt.plot(MSE[2:])
    plt.ylabel('MSE')
    plt.xlabel('times')
    plt.show()
    return time_count, w, MSE

def level_1_b(filename,learn_rate,method):
    data = pd.read_csv(filename, encoding='utf-8')
    X = np.array(data.iloc[:, 0:-1]) #提取数据集前列为X，此时X为N*D
    X = (X - np.mean(X, 0)) / (np.max(X, 0) - np.min(X, 0))# 特征零均值归一化，加快梯度下降速度
    x = np.vstack((np.ones(X.shape[0]), X.T)).T  # 在X左侧添加一列全为1的列
    y=np.array(data.iloc[:, -1]).reshape(X.shape[0], 1)  # 1*N
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    W = np.random.randn(X.shape[1] + 1, 1)
    if method == 'SGD':
        time_count, w, err = Stochastic_gradient_descent(x_train, y_train, W, learn_rate)
    else:
        time_count, w, err = Gradient_descent(x_train, y_train, W, learn_rate)
    MSE_Train = np.linalg.norm(np.dot(x_train, w) - y_train) ** 2 / x_train.shape[0]  # 训练误差
    MSE_Test = np.linalg.norm(y_test - np.dot(x_test, w)) ** 2 / x_test.shape[0]  # 预测误差
    print('train error:', MSE_Train)
    print('test error:', MSE_Test)
    print('W:', W)



#level_1_a("dataset_regression.csv")
#level_1_b("winequality-white.csv",0.1,'SGD')

#level_2_round1
# i=0
# data = pd.read_csv("winequality-white.csv", encoding='utf-8')
# X = np.array(data.iloc[:, 0:-1])  # 提取数据集前列为X，此时X为N*D
# X = (X - np.mean(X, 0)) / (np.max(X, 0) - np.min(X, 0))  # 特征零均值归一化，加快梯度下降速度
# x = np.vstack((np.ones(X.shape[0]), X.T)).T  # 在X左侧添加一列全为1的列
# y = np.array(data.iloc[:, -1]).reshape(X.shape[0], 1)  # 1*N
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# loss=[]
# #rates=np.arange(0.1,0.9,0.1)
# rates=[0.3,0.1,0.03,0.01,0.003,0.001]
# for learn_rate in rates:
#     W = np.random.randn(X.shape[1] + 1, 1)
#     MSE = []
#     MSE.append(100000)
#     MSE.append(10000)
#     time_count = 0
#     while np.abs(MSE[-1] - MSE[-2]) > 0.000001:
#         w_temp = np.zeros((W.shape[0], 1))
#         for j in range(W.shape[0]):  # 对所有的参数赋值
#             w_temp[j] = W[j] + learn_rate * np.dot((y_train - np.dot(x_train, W)).T, x_train[:, j]) / x_train.shape[0]
#         W = w_temp  # 更新参数
#         MSE.append(np.linalg.norm(np.dot(x_train, W) - y_train) ** 2 / x_train.shape[0])
#         # print(err[-1])
#         time_count += 1
#         if time_count>4000:
#             break
#
#     plt.plot(MSE[2:],label=("learn rate="+str(rates[i])))
#     i=i+1
#     # plt.savefig("level2_Figure"+str(i)+".png")
#     print('learn rate=', learn_rate)
#     MSE_Train = np.linalg.norm(np.dot(x_train, W) - y_train) ** 2 / x_train.shape[0]  # 训练误差
#     MSE_Test = np.linalg.norm(y_test - np.dot(x_test, W)) ** 2 / x_test.shape[0]  # 预测误差
#     print('train error:', MSE_Train)
#     print('test error:', MSE_Test)
#     print('W:', W)
#     loss.append(MSE_Train)
# plt.ylabel('MSE')
# plt.xlabel('times')
# plt.legend()
# plt.show()


# print(loss)
# print(rates)
# plt.plot(rates,loss)
# plt.show()

#level_3
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest_changelam(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)  # 标准化
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar  # （特征-均值）/方差
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):  # 测试不同的lambda取值，获得系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)  # 标准化
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar  # （特征-均值）/方差
    ws = ridgeRegres(xMat, yMat)
    return ws

data =  pd.read_csv("winequality-white.csv", encoding='utf-8')
X = np.array(data.iloc[:, 0:-1])
Y = np.array(data.iloc[:, -1])
ridgeWeights = ridgeTest_changelam(X, Y)
plt.plot(ridgeWeights)
plt.show()
w=ridgeTest(X, Y)
X = np.array(data.iloc[:, 0:-1]) #提取数据集前列为X，此时X为N*D
X = (X - np.mean(X, 0)) / np.var(X, 0)# 特征零均值归一化，加快梯度下降速度
y=np.array(data.iloc[:, -1]).reshape(X.shape[0], 1)  # 1*N
y=y-np.mean(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
MSE_Train = np.linalg.norm(np.dot(x_train, w) - y_train) ** 2 / x_train.shape[0]  # 训练误差
MSE_Test = np.linalg.norm(y_test - np.dot(x_test, w)) ** 2 / x_test.shape[0]  # 预测误差
print('train error:', MSE_Train)
print('test error:', MSE_Test)
print('W:', w)
