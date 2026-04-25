import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.pyplot import xlabel

#从cvs文件中读取数据存入x,y
data = pd.read_csv('customer.csv')
x = data['period']
y = data['avemoney']
print(x)
print(y)

#划分3个类别
sortColors =['r','g','b']

#初始化质心
sortX = np.random.randint(10,100,3)
sortY = np.random.randint(10,700,3)

#初始化类别为0
sortF = np.zeros(len(x), dtype=int)

#k-means
sortM = [[0,0,0],[0,0,0],[0,0,0]]  #分别代表x,y,个数
count = 10
while count > 0:
    sortM = [[0,0,0],[0,0,0],[0,0,0]]
    flag = 0 #标记类别
    for i in range(len(x)):
        min = 1000
        for j in range(3):
            distance = math.sqrt((x[i]-sortX[j])**2+(y[i]-sortY[j])**2)
            if distance < min:
                min = distance
                flag = j
                sortM[j][0] += x[i]
                sortM[j][1] += y[i]
                sortM[j][2] += 1
        sortF[i] = flag
    for i in range(3): #更新质心
        sortX[i] = sortM[i][0]//sortM[i][2]
        sortY[i] = sortM[i][1]//sortM[i][2]
    count -= 1

#可视化
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False  #中文与负号

plt.figure
plt.title('用户分类图')
plt.xlabel('平均消费周期（天）')
plt.ylabel('平均每次消费金额')

plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.yticks([100,150,200,250,300,350,400,450,500,550,600,650,700,750,800])

#显示3个中心点
for i in range(3):
    plt.scatter(sortX[i],sortY[i],marker='+',color=sortColors[i],label='1',s=30)

#显示其他点
for i in range(3):
    for j in range(len(x)):
        if sortF[j] == i:
            plt.scatter(x[j],y[j],marker='.',color=sortColors[i],label='1',s=50)
plt.show()