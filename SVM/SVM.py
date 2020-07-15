import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


#======================================================================
#                         载入数据及数据可视化
#======================================================================
data = sio.loadmat('data/ex6data1.mat')

X = data['X']
y = data['y']
print(X.shape,y.shape)

def plotData(X,y):
    pos_flag = y[:,0] == 1
    neg_flag = y[:,0] == 0
    pos = X[pos_flag]
    neg = X[neg_flag]
    plt.figure(figsize=(8,5))
    plt.scatter(pos[:,0],pos[:,1],marker='.',c='turquoise')
    plt.scatter(neg[:,0],neg[:,1],marker='x',c='darkorange')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)


#plotData(X,y)

#======================================================================
#                         画出决策边界
#======================================================================
def plotBoundary(clf,X):
    x1_min,x1_max = X[:,0].min(),X[:,0].max()
    x2_min,x2_max = X[:,1].min(),X[:,1].max()
    xx,yy = np.meshgrid(np.linspace(x1_min,x1_max,500),np.linspace(x2_min,x2_max,500))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx,yy,Z)

models = [svm.SVC(C,kernel='linear') for C in [1,100]]
clfs = [model.fit(X,y.ravel()) for model in models]
title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
for model,title in zip(clfs,title):
    plotData(X,y)
    plotBoundary(model,X) 
    plt.title(title)
    plt.show()
#当C比较大时模型对误分类的惩罚增大，比较严格，误分类少，间隔比较狭窄。

##========================================================
#                   非线性分类的高斯核SVM
##========================================================
#定义一个高斯核
def Gaussian_Kernel(x1,x2,sigma):
  return np.exp(-(np.sum((x1-x2)**2)/(2*(sigma**2))))

x1 = np.array([1.,2.,1.])
x2 = np.array([0.,4.,-1.])
sigma = 2
print(Gaussian_Kernel(x1,x2,sigma))


data2 = sio.loadmat('data/ex6data2.mat')
X2 = data2['X']
y2 = data2['y']
sigma = 0.1
gamma = np.power(sigma,-2.)/2
clf = svm.SVC(C=1,kernel='rbf',gamma=gamma)
model = clf.fit(X2,y2.ravel())
plotData(X2,y2)
plotBoundary(model,X2)
plt.show()

data3 = sio.loadmat('data/ex6data3.mat')
X3 = data3['X']
y3 = data3['y']
X3_val = data3['Xval']
y3_val = data3['yval']

Cv = (0.01,0.03,0.1,0.3,1,3,10)
sigmaV = Cv
best_pair,best_score = (0,0),0

for C in Cv:
  for sigma in sigmaV:
    gamma = np.power(sigma,-2.)/2
    model = svm.SVC(C=C,kernel='rbf',gamma=gamma)
    model.fit(X3,y3.ravel())
    cur_score = model.score(X3_val,y3_val)
    if cur_score > best_score:
      best_score = cur_score
      best_pair = (C,sigma)
print('best_pair={}, best_score={}'.format(best_pair, best_score))

model = svm.SVC(C=1,kernel='rbf',gamma = np.power(0.1,-2.)/2)
model.fit(X3,y3.ravel())
plotData(X3,y3)
plotBoundary(model,X3)

##========================================================
#                   Spam Classification
##========================================================
email = open('data/emailSample1.txt')
for line in open('data/emailSample1.txt'):
  print(line)

spam_train = sio.loadmat('data/spamTrain.mat')
spam_test = sio.loadmat('data/spamTest.mat')

X = spam_train['X']
X_test=spam_test['Xtest']
y = spam_train['y']
y_test = spam_test['ytest']

print(X.shape,X_test.shape,y.shape,y_test.shape)

clf = svm.SVC(C=0.1,kernel='linear')
clf.fit(X,y.ravel())
pred_train = clf.score(X,y)
pred_test = clf.score(X_test,y_test)
print(pred_train,pred_test)