# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

# 定义高斯函数，计算概率p(x|w)
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Gaussian_function(x, mean, cov):
    det_cov = np.linalg.det(cov)  # 计算方差矩阵的行列式
    inv_cov = np.linalg.inv(cov)  # 计算方差矩阵的逆
    # 计算概率p(x|w)
    p = 1 / (2 * np.pi * np.sqrt(det_cov)) * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean)))
    return p


# 生成正态分布数据
def Generate_Sample_Gaussian(mean, cov, P, label):
    '''
        mean 为均值向量
        cov 为方差矩阵a
        P 为单个类的先验概率
        return 单个类的数据集
    '''
    temp_num = round(1000 * P)
    x, y = np.random.multivariate_normal(mean, cov, temp_num).T
    z = np.ones(temp_num) * label
    X = np.array([x, y, z])
    # print(X)
    return X.T

def Generate_Sample_Gaussian_fortest(mean, cov, P, label):
    '''
        mean 为均值向量
        cov 为方差矩阵a
        P 为单个类的先验概率
        return 单个类的数据集
    '''
    temp_num = round(100 * P)
    x, y = np.random.multivariate_normal(mean, cov, temp_num).T
    z = np.ones(temp_num) * label
    X = np.array([x, y, z])
    # print(X)
    return X.T

# 根据不同先验生成不同的数据集
def Generate_DataSet(mean, cov, P):
    # 按照先验概率生成正态分布数据
    # 返回所有类的数据集
    X = []
    label = 1
    for i in range(3):
        # 把此时类i对应的数据集加到已有的数据集中
        X.extend(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    return X

def get_test_dataset(mean, cov, P):
    # 按照先验概率生成正态分布数据
    # 返回所有类的数据集
    X = []
    label = 1
    for i in range(3):
        # 把此时类i对应的数据集加到已有的数据集中
        X.extend(Generate_Sample_Gaussian_fortest(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    return X

def Generate_DataSet_plot(mean, cov, P):
    # 画出不同先验对应的散点图
    xx = []
    label = 1
    for i in range(3):
        xx.append(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    # 画图
    plt.figure()
    for i in range(3):
        plt.plot(xx[i][:, 0], xx[i][:, 1], '.', markersize=4.)
        plt.plot(mean[i][0], mean[i][1], 'r*')
    plt.show()
    return xx

# 似然率测试规则
def Likelihood_Test_Rule(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov)  # 计算样本i决策到j类的概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num


##最大后验概率规则
def Max_Posterior_Rule(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov) * P[j]  # 计算样本i是j类的后验概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i分到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num

def repeated_trials(mean, cov, P1, P2):
    # 根据mean，cov，P1,P2生成数据集X1,X2
    # 通过不同规则得到不同分类错误率并返回
    # 生成N=1000的数据集
    X1 = Generate_DataSet(mean, cov, P1)
    X2 = Generate_DataSet(mean, cov, P2)
    error = np.zeros((2, 2))
    # 计算似然率测试规则误差
    error_likelihood = Likelihood_Test_Rule(X1, mean, cov, P1)
    error_likelihood_2 = Likelihood_Test_Rule(X2, mean, cov, P2)
    error[0] = [error_likelihood, error_likelihood_2]
    # 计算最大后验概率规则误差
    error_Max_Posterior_Rule = Max_Posterior_Rule(X1, mean, cov, P1)
    error_Max_Posterior_Rule_2 = Max_Posterior_Rule(X2, mean, cov, P2)
    error[1] = [error_Max_Posterior_Rule, error_Max_Posterior_Rule_2]
    return error

def Gaussian_kernel_function(x,X,h):
    sum = 0
    num = np.array(X).shape[0]
    for i in range(num):
        sum += 1 / np.sqrt(2 * np.pi * h * h) * np.exp(-np.linalg.norm(x-X[i][0:2],ord=2) / (2*h*h))
    p=sum/num
    return p

# 核函数似然率测试规则
def Likelihood_kernel_Test_Rule(X, h,mode):
    L = random.sample(range(1, 999), 100)
    X_test=[]
    class_num = 3  # 类的个数
    X_train = [[], [], []]
    if mode == 1:
        X_test.extend(X[300:333])
        X_test.extend(X[633:666])
        X_test.extend(X[966:999])
        X_train[0] = X[0:300]
        X_train[1] = X[333:633]
        X_train[2] = X[666:966]
    else:
        X_test.extend(X[540:600])
        X_test.extend(X[870:900])
        X_test.extend(X[990:1000])
        X_train[0] = X[0:540]
        X_train[1] = X[600:870]
        X_train[2] = X[900:990]
    num = np.array(X_test).shape[0]
    P=[]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_kernel_function(X_test[i][0:2], X_train[j], h)  # 计算样本i决策到j类的概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
        P.append(p_temp[p_class-1])
        if p_class != X_test[i][2]:
            error_rate += 1
    A=np.array(X_test).T[0]
    B=np.array(X_test).T[1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    P=np.array(P)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')


    ax.plot_trisurf(A, B, P)

    plt.show()

    return error_rate / num

def repeated_trials_kernel(mean, cov, P1, P2):
    # 根据mean，cov，P1,P2生成数据集X1,X2
    # 通过不同规则得到不同分类错误率并返回
    # 生成N=1000的数据集
    X1 = Generate_DataSet(mean, cov, P1)
    X2 = Generate_DataSet(mean, cov, P2)
    # 计算似然率测试规则误差
    H=[0.1]
    for h in H:
        error_likelihood_kernel_1 = Likelihood_kernel_Test_Rule(X1,h,1)
        error_likelihood_kernel_2 = Likelihood_kernel_Test_Rule(X2,h,2)
        print(h,error_likelihood_kernel_1, error_likelihood_kernel_2)

# 找最邻近k点并计算p
def k_find(x,X,k):
    num=np.array(X).shape[0]
    dis=[]
    for i in range(num):
        dis.append(np.linalg.norm(x-X[i][0:2],ord=2))
    dis=np.array(dis)
    index=dis.argsort()
    d=np.zeros(k)
    point=[]
    for i in range(k):
        d[i]=dis[index[i]]
        point.append(X[index[i]])
    class_num=np.zeros(3)
    for i in range(k):
        if point[i][2] == 1:
            class_num[0]+=1
        if point[i][2] == 2:
            class_num[1]+=1
        if point[i][2] == 3:
            class_num[2]+=1
    V=np.pi*d[k-1]*d[k-1]
    p=class_num/(2*V)
    p_class = np.argmax(p) + 1
    return p_class

# k最近验证
def k_trials(mean, cov, P1, P2,k):
    X1 = Generate_DataSet(mean, cov, P1)
    X2 = Generate_DataSet(mean, cov, P2)
    X1_test = get_test_dataset(mean, cov, P1)
    X2_test = get_test_dataset(mean, cov, P2)

    error_rate_1=0
    error_rate_2=0
    num=np.array(X1_test).shape[0]
    for i in range(num):
        if X1_test[i][2] != k_find(X1_test[i][0:2],X1,k):
            error_rate_1 += 1
    error_rate_1=error_rate_1/num
    for i in range(num):
        if X2_test[i][2] != k_find(X2_test[i][0:2],X2,k):
            error_rate_2 += 1
    error_rate_2=error_rate_2/num
    print(k,error_rate_1,error_rate_2)



mean = np.array([[1, 1], [4, 4], [8, 1]])  # 均值数组
cov = [[2, 0], [0, 2]]  # 方差矩阵
num = 1000  # 样本个数
P1 = [1 / 3, 1 / 3, 1 / 3]  # 样本X1的先验概率
P2 = [0.6, 0.3, 0.1]  # 样本X2的先验概率
error_all = np.zeros((2, 2))

# #level_1
# # 测试times_num次求平均
# times_num = 10
# for times in range(times_num):
#     print(repeated_trials(mean,cov,P1,P2))
#     error_all += repeated_trials(mean, cov, P1, P2)
#     # 计算平均误差
# error_ave = error_all / times_num
# print(error_ave)

#level_2
repeated_trials_kernel(mean,cov,P1,P2)

# #level_3
# K=[1,3,5]
# for k in K:
#     k_trials(mean,cov,P1,P2,k)
