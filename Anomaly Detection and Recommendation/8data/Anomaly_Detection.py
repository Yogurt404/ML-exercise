import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from scipy import stats

##========================================================
#                       异常检测
##========================================================
data = sio.loadmat('8data/ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

def plotData(X):
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0],X[:,1])

def fitGaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu,sigma

mu,sigma = fitGaussian(X)#存放两个特征的拟合出来的均值和标准差,mu[0]代表第一个特征值的,mu[1]代表第二个特征值的
#  scipy.stats.norm函数 可以实现正态分布（也就是高斯分布）
p = np.zeros((X.shape[0],X.shape[1]))
# gauss = norm(loc=1, scale=2)  # loc: mean 均值， scale: standard deviation 标准差
p[:,0] = stats.norm.pdf(X[:,0],mu[0],sigma[0])#第一个特征值的高斯概率密度
p[:,1] = stats.norm.pdf(X[:,1],mu[1],sigma[1])#第二个特征值的高斯概率密度
print(p.shape)
#我们还需要为验证集（使用相同的模型参数）执行此操作。 
# 我们将使用与真实标签组合的这些概率来确定将数据点分配为异常的最佳概率阈值
pval = np.zeros((Xval.shape[0],Xval.shape[1]))
# gauss = norm(loc=1, scale=2)  # loc: mean 均值， scale: standard deviation 标准差
pval[:,0] = stats.norm.pdf(Xval[:,0],mu[0],sigma[0])#第一个特征值的高斯概率密度
pval[:,1] = stats.norm.pdf(Xval[:,1],mu[1],sigma[1])#第二个特征值的高斯概率密度
#阈值选择
def thresholdSelect(pval,yval):
    best_epsilon = 0 
    best_f1 = 0
    f1 = 0

    step = (pval.max()-pval.min())/1000#设置步长

    for epsilon in np.arange(pval.min(),pval.max(),step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds==1,yval==1)).astype(float)
        fp = np.sum(np.logical_and(preds==1,yval==0)).astype(float)
        fn = np.sum(np.logical_and(preds==0,yval==1)).astype(float)

        precision = tp/(tp+fp)#准确率 or 查准率
        recall = tp/(tp+fn)#召回率 or 查全率

        f1 = (2*precision*recall)/(precision+recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon,best_f1

epsilon,f1 = thresholdSelect(pval,yval)
print(epsilon,f1)

##=================================================
#                   可视化结果
##=================================================
result = np.where(p<epsilon)

plt.figure(figsize=(8,8))
plt.scatter(X[:,0],X[:,1])
plt.scatter(X[result[0],0],X[result[0],1],c='r')
plt.grid(True)
plt.show()




