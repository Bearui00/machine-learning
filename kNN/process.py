# _*_ encoding:GBK _*_

import numpy as np
import operator
import matplotlib.pyplot as plt
from numpy import array, zeros
from numpy.lib import tile
import sklearn
from sklearn.neighbors import KNeighborsClassifier

def run_lp(k,train_data):
    ls = len(train_data)
    correct_cnt=0
    for rec in range(ls):
        test_data=train_data[rec:rec+1]
        dataSet = np.vstack((train_data[0:rec],train_data[rec+1:ls]))
        neigh=KNeighborsClassifier(n_neighbors=k)
        neigh.fit(dataSet[...,:-1],dataSet[...,-1])
        predict_y=neigh.predict(test_data[...,:-1])
        if test_data[0][256]==predict_y[0]:
            correct_cnt+=1
    return correct_cnt

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def Img2Mat0(filename):
    f = open(filename)
    ss = f.readlines()
    l = len(ss)
    f.close()
    returnMat = zeros((l, 256))
    returnClVect = zeros((1, l))
    for i in range(l):
        sl = ss[i].split()
        for j in range(256):
            returnMat[i, j] = float(sl[j])
        Clcount = 0
        for j in range(256, 266):
            if sl[j] != '1':
                Clcount = Clcount + 1
            else:
                break
        returnClVect[0, i] = str(int(Clcount))
    return returnMat, returnClVect

def clsfy_loo(traData, traCls, k, rec=0):
    inX = traData[rec]
    ls = len(traData)
    dataSet = zeros((ls - 1, 256))
    labels = []
    j = 0
    for i in range(ls):
        if i != rec:
            dataSet[j] = traData[i]
            j = j + 1
            labels.append(str(int(traCls[0, i])))
    prCls = classify0(inX, dataSet, labels, k)
    if int(traCls[0, rec])!=int(prCls):
        return False
    else:
        return True

def clsfy_loo_cross(traData, traCls, k, testData):
    count=0
    len_test=len(testData)
    ls = len(traData)
    labels = []
    for i in range(ls):
        labels.append(str(int(traCls[0, i])))
    for j in range(len_test):
        prCls = classify0(testData[j], traData, labels, k)
        if int(testData[j][256])!=int(prCls):
            count+=1
    error_rate_per=count/len_test
    return error_rate_per



mm,kk=Img2Mat0('semeion.data')
data_all=np.hstack((mm,kk.T))
#中级要求
for k in [1,3,5]:
    count=run_lp(k,data_all)
    print(count/1593)


# #初级要求
# for k in [1,3,5]:
#     bingo = 0
#     for i in range(1592):
#         if clsfy_loo(mm, kk, k, i):
#             bingo = bingo + 1
#     rate = bingo / 1592
#     print(rate)



# #提高要求
# error_rate=[]
# for k in range(20):
#     rate=0
#     for i in range(5):
#         mm_test=data_all[i*300:i*300+299]
#         if i!=0:
#             mm_train=np.vstack((data_all[0:i*300-1],data_all[i*300+300:1592]))
#             kk_train=np.hstack((kk[:,0:i*300-1],kk[:,i*300+300:1592]))
#         else:
#             mm_train=data_all[300:1592]
#             kk_train=kk[:,300:1592]
#         rate_per=clsfy_loo_cross(mm_train,kk_train,k+1,mm_test)
#         rate+=rate_per
#     rate=rate/5
#     error_rate.append(rate)
# print(error_rate)
# k=np.arange(20)+1
# plt.plot(k,error_rate)
# plt.xlabel("k")
# plt.ylabel("error rate")
# plt.show()