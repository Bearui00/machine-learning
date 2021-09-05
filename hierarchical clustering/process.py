import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.optimize import linear_sum_assignment as linear_assignment
# 生成正态分布数据
def Generate_Sample_Gaussian(mean, cov, P, label):
    temp_num = round(2000 * P)
    x, y ,z= np.random.multivariate_normal(mean, cov, temp_num).T
    w = np.ones(temp_num) * label
    X = np.array([x, y, z,w])
    return X.T
# 根据不同先验生成不同的数据集
def Generate_DataSet(mean, cov, P):
    X = []
    label = 1
    for i in range(3):
        X.extend(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
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
    fig=plt.figure(1)
    ax=Axes3D(fig)
    for i in range(3):
        ax.scatter(xx[i][:, 0], xx[i][:, 1],xx[i][:,2])
        ax.scatter(mean[i][0], mean[i][1],mean[i][2])
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    return xx

def euler_distance(point1: np.ndarray, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

class ClusterNode(object):
    def __init__(self, vec, left=None, right=None, distance=-1, id=None, count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
        :param left: 左节点
        :param right:  右节点
        :param distance: 两个节点的距离
        :param id: 用来标记哪些节点是计算过的
        :param count: 这个节点的叶子节点个数
        """
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count

class Hierarchical(object):
    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None
    def fit(self, x):
        nodes = [ClusterNode(vec=v, id=i) for i,v in enumerate(x)]
        distances = {}
        point_num, future_num = np.shape(x)  # 特征的维度
        self.labels = [ -1 ] * point_num
        self.idlist = [ 0 ] * point_num
        currentclustid = -1
        while len(nodes) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None  # 表示最相似的两个聚类
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances:
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            # 合并两个聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_vec = [ (node1.vec[i] * node1.count + node2.vec[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1
            del nodes[part2], nodes[part1]   # 一定要先del索引较大的
            nodes.append(new_node)
        self.nodes = nodes
        self.calc_label_id()

    def calc_label_id(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes,start=1):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点
        """
        if node.left == None and node.right == None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

def clustering_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    su = 0
    # print(ind[0].shape[0])
    for t in range(ind[0].shape[0]):
        su += w[ind[0][t], ind[1][t]]
    su = su * 1.0 / y_pred.size
    return su

class SingleLinkage:

    def __init__(self, data, k):
        self.k = k
        self.data = data
        self.fit()

    def fit(self):
        n = len(self.data)
        self.clusters = {}
        for i in range(n):
          self.clusters[i] = []
          self.clusters[i].append(i)
        self.dist = np.sqrt((np.square(self.data[:,np.newaxis]-self.data).sum(axis=2)))
        for i in range(n-self.k):
            merge = self.merging()
            self.clusters[merge[0]] = self.clusters[merge[0]] + self.clusters[merge[1]]
            self.clusters.pop(merge[1])
            for j in range(n):
                if self.dist[j,merge[0]]>self.dist[j,merge[1]]:
                    self.dist[j,merge[0]]=self.dist[j,merge[1]]
        for i in range(self.k):
            while not i in self.clusters:
                for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:
                    self.clusters[j-1] = self.clusters.pop(j)
        for i in self.clusters.keys():
            self.clusters[i].sort()

    def merging(self):
        mini = 1e99
        merge = (None, None)
        for i in list(map(int, self.clusters.keys())):
            for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:
                if self.dist[i][j] < mini:
                    mini = self.dist[i][j]
                    merge = (i, j)
        return merge

class CompleteLinkage:

    def __init__(self, data, k):
        self.k = k
        self.data = data
        self.fit()

    def fit(self):
        n = len(self.data)
        self.clusters = {}
        for i in range(n):
          self.clusters[i] = []
          self.clusters[i].append(i)
        self.dist = np.sqrt((np.square(self.data[:,np.newaxis]-self.data).sum(axis=2)))
        for i in range(n-self.k):
            merge = self.merging()
            print(merge)
            self.clusters[merge[0]] = self.clusters[merge[0]] + self.clusters[merge[1]]
            self.clusters.pop(merge[1])
            for j in range(n):
                if self.dist[j,merge[0]]<self.dist[j,merge[1]]:
                    self.dist[j,merge[0]]=self.dist[j,merge[1]]
        for i in range(self.k):
            while not i in self.clusters:
                for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:
                    self.clusters[j-1] = self.clusters.pop(j)
        for i in self.clusters.keys():
            self.clusters[i].sort()

    def merging(self):
        mini = 1e99 
        merge = (None, None)
        for i in list(map(int, self.clusters.keys())):
            for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:

                if self.dist[i][j] < mini:
                    mini = self.dist[i][j]
                    merge = (i, j)
        return merge

if __name__=='__main__':
    # 用于产生聚类的中心点, 聚类中心的维度代表产生样本的维度
    # centers=[[1,1,1],[3,4,3],[7,6,5],[1,5,7]]
    # 产生用于聚类的数据集，聚类中心点的个数代表类别数
    # X,labels_true=gendata(centers,1000,0.5)
    mean = np.array([[1, 1, 1], [4, 4, 4], [8, 8, 1]])  # 均值数组
    cov = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 方差矩阵
    P1 = [1 / 3, 1 / 3, 1 / 3]  # 样本X1的先验概率
    # Generate_DataSet_plot(mean, cov, P1)  # 画X1数据集散点图
    X = Generate_DataSet(mean, cov, P1)
    X=np.array(X)
    print(X)
    labels_true=np.array(X[:,-1])
    X=np.array(X[:,0:3])
    row=X.shape[0]

    # Single
    # hc = SingleLinkage(X, 3)
    # print(hc.clusters)
    # labels_pred=[]
    # a=0
    # b=0
    # c=0
    # for i in range(row):
    #     if a<len(hc.clusters[0]) and hc.clusters[0][a]==i:
    #         labels_pred.append(0)
    #         a=a+1
    #
    #     if b<len(hc.clusters[1]) and hc.clusters[1][b]==i:
    #         labels_pred.append(1)
    #         b=b+1
    #
    #     if c<len(hc.clusters[2]) and hc.clusters[2][c]==i:
    #         labels_pred.append(2)
    #         c=c+1
    # labels=np.array(labels_pred).T
    # labels += 1
    # X_1=np.column_stack((X,labels))
    # xx=[]
    # xx_1=[]
    # xx_2=[]
    # xx_3=[]
    # for each in X_1:
    #     if each[3]==1:
    #         xx_1.append(each)
    #     elif each[3]==2:
    #         xx_2.append(each)
    #     else:
    #         xx_3.append(each)
    # xx_1=np.array(xx_1)
    # xx_2 = np.array(xx_2)
    # xx_3 = np.array(xx_3)
    # xx.append(xx_1)
    # xx.append(xx_2)
    # xx.append(xx_3)
    # print(xx)
    # fig = plt.figure(2)
    # ax = Axes3D(fig)
    # for i in range(3):
    #     ax.scatter(xx[i][:, 0], xx[i][:, 1], xx[i][:, 2])
    #     ax.scatter(mean[i][0], mean[i][1], mean[i][2])
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()

    # Complete
    # hc1 = CompleteLinkage(X, 3)
    # print(hc1.clusters)
    # labels_pred1 = []
    # a = 0
    # b = 0
    # c = 0
    # for i in range(row):
    #     if a < len(hc1.clusters[0]) and hc1.clusters[0][a] == i:
    #         labels_pred1.append(0)
    #         a = a + 1
    #
    #     if b < len(hc1.clusters[1]) and hc1.clusters[1][b] == i:
    #         labels_pred1.append(1)
    #         b = b + 1
    #
    #     if c < len(hc1.clusters[2]) and hc1.clusters[2][c] == i:
    #         labels_pred1.append(2)
    #         c = c + 1
    # labels1 = np.array(labels_pred1).T
    # labels1 += 1
    # X_2 = np.column_stack((X, labels1))
    # xx1 = []
    # xx1_1 = []
    # xx1_2 = []
    # xx1_3 = []
    # for each in X_2:
    #     if each[3] == 1:
    #         xx1_1.append(each)
    #     elif each[3] == 2:
    #         xx1_2.append(each)
    #     else:
    #         xx1_3.append(each)
    # xx1_1 = np.array(xx1_1)
    # xx1_2 = np.array(xx1_2)
    # xx1_3 = np.array(xx1_3)
    # xx1.append(xx1_1)
    # xx1.append(xx1_2)
    # xx1.append(xx1_3)
    # print(xx1)
    # fig = plt.figure(3)
    # ax = Axes3D(fig)
    # for i in range(3):
    #     ax.scatter(xx1[i][:, 0], xx1[i][:, 1], xx1[i][:, 2])
    #     ax.scatter(mean[i][0], mean[i][1], mean[i][2])
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()

    #Average
    my1=Hierarchical(3)
    my1.fit(X)
    print(np.array(my1.labels))
    labels2 = np.array(my1.labels).T
    X_3 = np.column_stack((X, labels2))
    xx2 = []
    xx2_1 = []
    xx2_2 = []
    xx2_3 = []
    for each in X_3:
        if each[3] == 1:
            xx2_1.append(each)
        elif each[3] == 2:
            xx2_2.append(each)
        else:
            xx2_3.append(each)
    xx2_1 = np.array(xx2_1)
    xx2_2 = np.array(xx2_2)
    xx2_3 = np.array(xx2_3)
    xx2.append(xx2_1)
    xx2.append(xx2_2)
    xx2.append(xx2_3)
    print(xx2)
    fig = plt.figure(4)
    ax = Axes3D(fig)
    for i in range(3):
        ax.scatter(xx2[i][:, 0], xx2[i][:, 1], xx2[i][:, 2])
        ax.scatter(mean[i][0], mean[i][1], mean[i][2])
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    a=clustering_acc(np.array(X[:,-1]),np.array(my1.labels))
    print(a)



            
            
    
