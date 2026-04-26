import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
data = pd.read_csv('customer.csv')
x = data['period']
y = data['avemoney']
points = np.column_stack((x, y))

#k-means++初始化质心
def kmeans_plusplus_init(points,k):
    n = len(points)
    centers = []
    #随机选择第一个质心
    k1_index = np.random.randint(n)
    centers.append(points[k1_index])

    #选择剩余执行难
    for _ in range(1,k):
        #计算每个点到最近中心的距离
        #np.linalg.norm(p - c)：计算点p和某个中心点c之间的欧几里得距离
        #distances 数组中的每个元素 distances[i] 代表了 points[i] 这个点到它最近的那个中心点的距离。
        distances = np.array([min(np.linalg.norm(p - c)for c in centers)for p in points])
        # distances = []
        # for i in range(len(points)):
        #     d = []
        #     for j in range(len(centers)):
        #         d.append(np.sqrt((points[i].x-centers[j].x)**2 + (points[i].y-centers[j].y)**2))
        #     distances.append(min(d))

        #距离平方作为概率
        probs = distances ** 2
        probs /= np.sum(probs)

        #按概率选择下一质心
        idx = np.random.choice(n, p=probs)
        centers.append(points[idx])

    return np.array(centers)

#k-means主函数
def kmeans(points, k=3, max_iter=100, tol=1e-4): #max_iter：最大迭代次数（默认值：100） ；tol：容忍度（收敛阈值）； 1e-4：10的负4次方，即0.0001
    # 初始化质心
    centers = kmeans_plusplus_init(points, k)
    labels = np.zeros(len(points), dtype=int)

    for _ in range(max_iter):
        # 分配样本到最近质心
        distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2) #axis=2意为指定沿着最后一个维度（特征维度 d，即坐标）进行计算
        #使用 np.linalg.norm(points[:, np.newaxis] - centers, axis=2)，NumPy 会在底层利用 C 语言和 SIMD 指令集并行处理所有计算，速度通常比 Python 循环快 几十到几百倍
        new_labels = np.argmin(distances, axis=1) #argmin返回最小值的索引（下标）

        # 检查是否收敛
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # 更新质心
        new_centers = np.zeros_like(centers)# np.zeros_like(centers) 会创建一个和 centers 形状、类型完全一样的全0矩阵
        for i in range(k):

            # 【关键技巧】布尔索引：筛选出所有属于第 i 个簇的点
            # labels == i 会生成一个布尔数组（如 [True, False, True...]）
            # points[...] 会只取出对应位置为 True 的行
            cluster_points = points[labels == i]

            # 如果随机选到了一个远离所有数据聚集区的离群点作为初始中心，可能没有任何数据点认为它是“最近”的中心。引发空簇；或者设置的质心过多也可能空簇
            # 处理“空簇”现象：如果某个簇没有分配到任何点（len == 0），直接求均值会报错
            if len(cluster_points) == 0:
                # 空簇处理：重新随机选一个点，防止算法崩溃
                new_centers[i] = points[np.random.choice(len(points))]
            else:
                new_centers[i] = cluster_points.mean(axis=0)

        # 检查质心变化
        if np.linalg.norm(new_centers - centers) < tol: #计算新旧中心之间的距离（移动量）
            centers = new_centers
            break
        centers = new_centers

    return labels, centers

#聚类
labels, final_centers = kmeans(points, k=3)

#可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))  # 设置画布大小

plt.title('用户分类图 (K-Means++)')
plt.xlabel('平均消费周期（天）')
plt.ylabel('平均每次消费金额')

# 设置坐标轴刻度
plt.xticks(range(10, 101, 10))
plt.yticks(range(100, 801, 50))

# 定义颜色和标记
colors = ['r', 'g', 'b']  # 对应簇1, 簇2, 簇3

# 1. 绘制数据点 (利用 labels 进行筛选)
for i in range(3):
    # 筛选出属于当前簇 i 的所有点
    cluster_points = points[labels == i]

    # 绘制这些点，并指定颜色
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f'簇{i + 1}', s=50, alpha=0.6)

# 2. 绘制质心 (使用 final_centers)
plt.scatter(final_centers[:, 0], final_centers[:, 1],
            c='black', marker='X', s=20, label='质心', edgecolors='white')

# 添加图例和网格
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()