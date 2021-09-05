# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
import re
# 定义判断结点形状,其中boxstyle表示文本框类型,fc指的是注释框颜色的深度
decisionNode = dict(boxstyle="round4", color='r', fc='0.9')
# 定义叶结点形状
leafNode = dict(boxstyle="circle", color='m')
# 定义父节点指向子节点或叶子的箭头形状
arrow_args = dict(arrowstyle="<-", color='g')


def plot_node(node_txt, center_point, parent_point, node_style):
    '''
    绘制父子节点，节点间的箭头，并填充箭头中间上的文本
    :param node_txt:文本内容
    :param center_point:文本中心点
    :param parent_point:指向文本中心的点
    '''
    createPlot.ax1.annotate(node_txt,
                            xy=parent_point,
                            xycoords='axes fraction',
                            xytext=center_point,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=node_style,
                            arrowprops=arrow_args)


def get_leafs_num(tree_dict):
    '''
    获取叶节点的个数
    :param tree_dict:树的数据字典
    :return tree_dict的叶节点总个数
    '''
    # tree_dict的叶节点总数
    leafs_num = 0

    # 字典的第一个键，也就是树的第一个节点
    root = list(tree_dict.keys())[0]
    # 这个键所对应的值，即该节点的所有子树。
    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        # 检测子树是否字典型
        if type(child_tree_dict[key]).__name__ == 'dict':
            # 子树是字典型，则当前树的叶节点数加上此子树的叶节点数
            leafs_num += get_leafs_num(child_tree_dict[key])
        else:
            # 子树不是字典型，则当前树的叶节点数加1
            leafs_num += 1

    # 返回tree_dict的叶节点总数
    return leafs_num


def get_tree_max_depth(tree_dict):
    '''
    求树的最深层数
    :param tree_dict:树的字典存储
    :return tree_dict的最深层数
    '''
    # tree_dict的最深层数
    max_depth = 0

    # 树的根节点
    root = list(tree_dict.keys())[0]
    # 当前树的所有子树的字典
    child_tree_dict = tree_dict[root]

    for key in child_tree_dict.keys():
        # 树的当前分支的层数
        this_path_depth = 0
        # 检测子树是否字典型
        if type(child_tree_dict[key]).__name__ == 'dict':
            # 如果子树是字典型，则当前分支的层数需要加上子树的最深层数
            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])
        else:
            # 如果子树不是字典型，则是叶节点，则当前分支的层数为1
            this_path_depth = 1
        if this_path_depth > max_depth:
            max_depth = this_path_depth

    # 返回tree_dict的最深层数
    return max_depth


def plot_mid_text(center_point, parent_point, txt_str):
    '''
    计算父节点和子节点的中间位置，并在父子节点间填充文本信息
    :param center_point:文本中心点
    :param parent_point:指向文本中心点的点
    '''

    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    createPlot.ax1.text(x_mid, y_mid, txt_str)
    return


def plotTree(tree_dict, parent_point, node_txt):
    '''
    绘制树
    :param tree_dict:树
    :param parent_point:父节点位置
    :param node_txt:节点内容
    '''

    leafs_num = get_leafs_num(tree_dict)
    root = list(tree_dict.keys())[0]
    # plotTree.totalW表示树的深度
    center_point = (plotTree.xOff + (1.0 + float(leafs_num)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 填充node_txt内容
    plot_mid_text(center_point, parent_point, node_txt)
    # 绘制箭头上的内容
    plot_node(root, center_point, parent_point, decisionNode)
    # 子树
    child_tree_dict = tree_dict[root]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 因从上往下画，所以需要依次递减y的坐标值，plotTree.totalD表示存储树的深度
    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            plotTree(child_tree_dict[key], center_point, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(child_tree_dict[key], (plotTree.xOff, plotTree.yOff), center_point, leafNode)
            plot_mid_text((plotTree.xOff, plotTree.yOff), center_point, str(key))
    # h绘制完所有子节点后，增加全局变量Y的偏移
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

    return


def createPlot(tree_dict):
    '''
    绘制决策树图形
    :param tree_dict
    :return 无
    '''
    # 设置绘图区域的背景色
    fig = plt.figure(1, facecolor='white')
    # 清空绘图区域
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig.clf()
    # 定义横纵坐标轴,注意不要设置xticks和yticks的值!!!
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 由全局变量createPlot.ax1定义一个绘图区，111表示一行一列的第一个，frameon表示边框,**axprops不显示刻度
    plotTree.totalW = float(get_leafs_num(tree_dict))
    plotTree.totalD = float(get_tree_max_depth(tree_dict))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree_dict, (0.5, 1.0), '')
    plt.show()


#计算熵
def calcEntropy(dataSet):
    mD = len(dataSet)
    dataLabelList = [x[-1] for x in dataSet]
    dataLabelSet = set(dataLabelList)
    ent = 0
    for label in dataLabelSet:
        mDv = dataLabelList.count(label)
        prop = float(mDv) / mD
        ent = ent - prop * np.math.log(prop, 2)
    return ent

# 拆分数据集
# index - 要拆分的特征的下标
# feature - 要拆分的特征
# 返回值 - dataSet中index所在特征为feature，且去掉index一列的集合
def splitDataSet(dataSet, index, feature):
    splitedDataSet = []
    for data in dataSet:
        if(data[index] == feature):
            sliceTmp = data[:index]
            sliceTmp.extend(data[index + 1:])
            splitedDataSet.append(sliceTmp)
    return splitedDataSet

#计算信息增益-选择最好的特征-返回下标
def chooseBestFeature(dataSet):
    entD = calcEntropy(dataSet)
    mD = len(dataSet)
    featureNumber = len(dataSet[0]) - 1
    maxGain = -100
    maxIndex = -1
    for i in range(featureNumber):
        entDCopy = entD
        featureI = [x[i] for x in dataSet]
        featureSet = set(featureI)
        # print(featureSet)
        for feature in featureSet:
            splitedDataSet = splitDataSet(dataSet, i, feature)
            # print(splitedDataSet)
            mDv = len(splitedDataSet)
            entDCopy = entDCopy - float(mDv) / mD * calcEntropy(splitedDataSet)
        if(maxIndex == -1):
            maxGain = entDCopy
            maxIndex = i
        elif(maxGain < entDCopy):
            maxGain = entDCopy
            maxIndex = i
    return maxIndex

# 寻找最多的，作为标签
def mainLabel(labelList):
    labelRec = labelList[0]
    maxLabelCount = -1
    labelSet = set(labelList)
    for label in labelSet:
        if(labelList.count(label) > maxLabelCount):
            maxLabelCount = labelList.count(label)
            labelRec = label
    return labelRec

#生成树
def createDecisionTree_ID3(dataSet, featureNames):
    labelList = [x[-1] for x in dataSet]
    if(len(dataSet[0]) == 1): #没有可划分的属性了
        return mainLabel(labelList)  #选出最多的label作为该数据集的标签
    elif(labelList.count(labelList[0]) == len(labelList)): # 全部都属于同一个Label
        return labelList[0]
    bestFeatureIndex = chooseBestFeature(dataSet)
    bestFeatureName = featureNames.pop(bestFeatureIndex)
    myTree = {bestFeatureName: {}}
    featureList = [x[bestFeatureIndex] for x in dataSet]
    featureSet = set(featureList)
    for feature in featureSet:
        featureNamesNext = featureNames[:]
        splitedDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)
        myTree[bestFeatureName][feature] = createDecisionTree_ID3(splitedDataSet, featureNamesNext)
    return myTree

def splitDataSet_c(dataSet, index, value, LorR='L'):
    splitedDataSet  = []
    if LorR == 'L':
        for data in dataSet:
            if float(data[index]) < value:
                splitedDataSet .append(data)
    else:
        for data in dataSet:
            if float(data[index]) > value:
                splitedDataSet .append(data)
    return splitedDataSet

# 选择最好的数据集划分方式
def chooseBestFeature_c45(dataSet, labelProperty):
    featureNumber = len(labelProperty)
    entD = calcEntropy(dataSet)
    maxGain = -100
    maxIndex = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    for i in range(featureNumber):  # 对每个特征循环
        featurel = [x[i] for x in dataSet]
        featurelSet = set(featurel)  # 该特征包含的所有值
        entN = 0.0
        bestPartValuei = None
        if labelProperty[i] == 0:  # 对离散的特征
            for feature in featurelSet:
                splitedDataSet = splitDataSet(dataSet, i, feature)
                prob = len(splitedDataSet) / float(len(dataSet))
                entN += prob * calcEntropy(splitedDataSet)
        else:  # 对连续的特征
            sortedUniqueVals = list(featurelSet)  # 对特征值排序
            sortedUniqueVals.sort()
            entM = 1e5
            for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # 对每个划分点，计算信息熵
                dataSetLeft = splitDataSet_c(dataSet, i, partValue, 'L')
                dataSetRight = splitDataSet_c(dataSet, i, partValue, 'R')
                probLeft = len(dataSetLeft) / float(len(dataSet))
                probRight = len(dataSetRight) / float(len(dataSet))
                Entropy = probLeft * calcEntropy(
                    dataSetLeft) + probRight * calcEntropy(dataSetRight)
                if Entropy < entM:  # 取最小的信息熵
                    entM = Entropy
                    bestPartValuei = partValue
            entN = entM
        infoGain = entD - entN  # 计算信息增益
        if infoGain > maxGain:  # 取最大的信息增益对应的特征
            maxGain = infoGain
            maxIndex = i
            bestPartValue = bestPartValuei
    return maxIndex, bestPartValue

# 创建树, 样本集 特征 特征属性[0 离散， 1 连续]
def createDecisionTree_c45(dataSet, labels, labelProperty):
    labelList = [x[-1] for x in dataSet]
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return mainLabel(labelList)
    bestFeatureIndex, bestPartValue = chooseBestFeature_c45(dataSet,labelProperty)
    if bestFeatureIndex == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        return mainLabel(labelList)
    if labelProperty[bestFeatureIndex] == 0:  # 对离散的特征
        bestFeatureName = labels[bestFeatureIndex]
        myTree = {bestFeatureName: {}}
        featureNames = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (featureNames[bestFeatureIndex])  # 已经选择的特征不再参与分类
        del (labelPropertyNew[bestFeatureIndex])
        featureList = [x[bestFeatureIndex] for x in dataSet]
        featureSet = set(featureList)  # 该特征包含的所有值
        for feature in featureSet:  #递归
            featureNamesNext = featureNames[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatureName][feature] = createDecisionTree_c45(
                splitDataSet(dataSet, bestFeatureIndex, feature), featureNamesNext,
                subLabelProperty)
    else:  # 对连续的特征，不删除该特征，分别构建左子树和右子树
        bestFeatureName = labels[bestFeatureIndex] + '<' + str(bestPartValue)
        myTree = {bestFeatureName: {}}
        featureNamesNext = labels[:]
        subLabelProperty = labelProperty[:]
        # 构建左子树
        valueLeft = '是'
        myTree[bestFeatureName][valueLeft] = createDecisionTree_c45(
            splitDataSet_c(dataSet, bestFeatureIndex, bestPartValue, 'L'), featureNamesNext,
            subLabelProperty)
        # 构建右子树
        valueRight = '否'
        myTree[bestFeatureName][valueRight] = createDecisionTree_c45(
            splitDataSet_c(dataSet, bestFeatureIndex, bestPartValue, 'R'), featureNamesNext,
            subLabelProperty)
    return myTree


# 测试算法
def classify_c(inputTree, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    lessIndex = str(firstStr).find('<')
    if lessIndex > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # 跟节点对应的特征
    classLabel = None
    for key in secondDict.keys():  # 对每个分支循环
        if featLabelProperties[featIndex] == 0:  # 离散的特征 ->同ID3遍历字典
            if testVec[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict[key], featLabels,featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict[key]
        else:
            partValue = float(str(firstStr)[lessIndex + 1:])
            if testVec[featIndex] < partValue:  # 进入左子树
                if type(secondDict['是']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict['是'], featLabels,
                                            featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict['是']
            else:
                if type(secondDict['否']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict['否'], featLabels,
                                            featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict['否']
    return classLabel
# ID3
data = pd.read_csv("Watermelon-train1.csv",encoding='gbk')
data.drop('编号',axis=1,inplace=True)
labels=data.columns.values.tolist()
dataset=data.values.tolist()
k=createDecisionTree_ID3(dataset,labels)
print(k)
createPlot(k)
test=pd.read_csv("Watermelon-test1.csv")
print(k.keys(),k.values())
lenth=len(test)
correct=0
for i in range(lenth):
    kd = k
    result=''
    while 1:
        att = list(kd.keys())[0]
        result = test.at[i, att]
        kd = list(kd.values())
        kd = kd[0]
        kd = kd[result]
        if kd == '是' or kd=='否':
            break
    print(kd)
    if kd == test.at[i,'好瓜']:
        correct += 1
print(correct)
print(correct/lenth)



# # C4.5
# data = pd.read_csv("Watermelon-train2.csv",encoding='gbk')
# data.drop('编号',axis=1,inplace=True)
# dataset=data.values.tolist()
# print(data)
# labels=['色泽', '根蒂', '敲声', '纹理', '密度']
# print(dataset)
# labelpro=[0,0,0,0,1]
# tree=createDecisionTree_c45(dataset, labels, labelpro)
# print(tree)
# createPlot(tree)
# test = pd.read_csv("Watermelon-test2.csv",encoding='gbk')
# test.drop('编号',axis=1,inplace=True)
# testset=test.values.tolist()
# correct=0
# for i in range(len(testset)):
#     res=classify_c(tree,labels,labelpro,testset[i])
#     print(res)
#     if res == testset[i][-1]:
#         correct += 1
# print(correct)
# print(correct/len(testset))
