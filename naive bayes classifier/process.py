import pandas as pd
import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

TT=[]
label1=[]
label2=[]
label3=[]
# get p(x|ck)
def get_p(x_data,c,mean,var):
    p=1
    for i in range(13):
        p *= norm.pdf(x=x_data[i+1], loc=mean[c][i], scale = np.sqrt(var[c][i]))
    return p

def get_c(test,C,mean,var):
    temp=[]
    for i in range(3):
        temp.append(get_p(test,i,mean,var)*C[i])
    p_temp=np.argmax(temp)+1
    print(temp)
    TT.append(temp)
    return p_temp

#59:71:48->60:72:48=5:6:4,volume:30
def devide(data):
    D = [[], [], []]
    test=[]
    N1 = list(range(0, 59))
    L1 = random.sample(range(0, 59), 10)
    for i in L1:
        if i in N1:
            N1.remove(i)
    for i in N1:
        D[0].append(data[i])
    for i in L1:
        test.append((data[i]))
    N2 = list(range(59, 130))
    L2 = random.sample(range(59, 130), 12)
    for i in L2:
        if i in N2:
            N2.remove(i)
    for i in N2:
        D[1].append(data[i])
    for i in L2:
        test.append((data[i]))
    N3 = list(range(130, 178))
    L3 = random.sample(range(130, 178), 8)
    for i in L3:
        if i in N3:
            N3.remove(i)
    for i in N3:
        D[2].append(data[i])
    for i in L3:
        test.append((data[i]))
    return D,test

def draw_confusion(TP,FP,FN,TN):
    confusion = np.array(([TP, FP], [FN, TN]))
    plt.imshow(confusion)
    indices = range(len(confusion))
    plt.xticks(indices, ['Class', 'Others'])
    plt.yticks(indices, ['Class', 'Others'])
    plt.colorbar()
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.title('Confusion')
    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.show()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = 2 / (1 / precision + 1 / recall)
    print("精度：%f 召回率：%f F值：%f" % (precision, recall, F))

data=pd.read_csv("wine.data",header=None)
data=np.array(data)
D,test=devide(data)
C=[49/148,59/148,40/148]
mean=[[],[],[]]
var=[[],[],[]]
for c in range(3):
    for i in range(13):
        mean[c].append(np.mean(np.array(D[c]).T[i+1]))
        var[c].append(np.var(np.array(D[c]).T[i+1]))
error=0
p1,p2,p3=0,0,0
T=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(30):
    A=get_c(test[i],C,mean,var)
    T[A-1][int(test[i][0])-1] += 1
    if test[i][0] != A:
        error +=1
        print(A,test[i][0],error)
    if  test[i][0] == 1: #2/3
        l=[1,TT[i][0]]
        label1.append(l)
        p1+=1
    else:
        l=[0,TT[i][0]]
        label1.append(l)
    if  test[i][0] == 2: #2/3
        l=[1,TT[i][1]]
        label2.append(l)
        p2+=1
    else:
        l=[0,TT[i][1]]
        label2.append(l)
    if  test[i][0] == 3: #2/3
        l=[1,TT[i][2]]
        label3.append(l)
        p3+=1
    else:
        l=[0,TT[i][2]]
        label3.append(l)
print(error/30)
TP_1=T[0][0]
FP_1=T[0][1]+T[0][2]
FN_1=T[1][0]+T[2][0]
TN_1=T[1][1]+T[1][2]+T[2][1]+T[2][2]
TP_2=T[1][1]
FP_2=T[1][0]+T[1][2]
FN_2=T[0][1]+T[2][1]
TN_2=T[0][0]+T[0][2]+T[2][0]+T[2][2]
TP_3=T[2][2]
FP_3=T[2][0]+T[2][1]
FN_3=T[0][2]+T[1][2]
TN_3=T[0][0]+T[1][0]+T[1][1]+T[0][1]
draw_confusion(TP_1,FP_1,FN_1,TN_1)

draw_confusion(TP_2,FP_2,FN_2,TN_2)

draw_confusion(TP_3,FP_3,FN_3,TN_3)


#ROC1
label1=sorted(label1,key=lambda x:x[1],reverse=True)
point1=[]
for i in range(30):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    for j in range(i+1):
        if label1[j][0]==1:
            TP1+=1
        else:
            FP1+=1
    FN=p1-TP1
    TN=30-p1-FP1
    FPR=FP1/(FP1+TN)
    TPR=TP1/p1
    point1.append([FPR,TPR])
point=np.array(point1)
x=point[:,0]
y=point[:,1]
temp=[0,1]
# x_smooth=np.linspace(x.min(),x.max(),300)
# y_smooth=make_interp_spline(x,y)(x_smooth)
# plt.plot(x_smooth,y_smooth,'r', markersize=4.)
plt.plot(x,y,'r', markersize=4.)
plt.plot(temp,temp,'b', markersize=4.)
plt.show()
print(np.trapz(y,x))


#ROC2
label2=sorted(label2,key=lambda x:x[1],reverse=True)
point2=[]
for i in range(30):
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for j in range(i+1):
        if label2[j][0]==1:
            TP2+=1
        else:
            FP2+=1
    FN=p2-TP2
    TN=30-p2-FP2
    FPR=FP2/(FP2+TN)
    TPR=TP2/p2
    point2.append([FPR,TPR])
point=np.array(point2)
x=point[:,0]
y=point[:,1]
temp=[0,1]
plt.plot(x,y,'r', markersize=4.)
plt.plot(temp,temp,'b', markersize=4.)
plt.show()
print(np.trapz(y,x))

#ROC2
label3=sorted(label3,key=lambda x:x[1],reverse=True)
point3=[]
for i in range(30):
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    for j in range(i+1):
        if label3[j][0]==1:
            TP3+=1
        else:
            FP3+=1
    FN=p3-TP3
    TN=30-p3-FP3
    FPR=FP3/(FP3+TN)
    TPR=TP3/p3
    point3.append([FPR,TPR])
point=np.array(point3)
x=point[:,0]
y=point[:,1]
temp=[0,1]
plt.plot(x,y,'r', markersize=4.)
plt.plot(temp,temp,'b', markersize=4.)
plt.show()
print(np.trapz(y,x))