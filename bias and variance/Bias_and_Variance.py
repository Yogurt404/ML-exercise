import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd

#======================================================================
#                           载入数据
#======================================================================

data =sio.loadmat('ex5data1.mat')
# Training set
X = data['X']
y = data['y']
# Cross Validation
Xval = data['Xval']
yval = data['yval']
# Test set
Xtest = data['Xtest']
ytest = data['ytest']

print(X.shape,y.shape,Xval.shape,yval.shape,Xtest.shape,ytest.shape)

#plt.figure(figsize=(8,5))
#plt.scatter(X,y,c='r',marker = 'x')
#plt.grid(True)
#plt.show()
def bias(X):
    X0 = np.ones(X.shape[0])
    return X0
X = np.c_[bias(X),X]
Xval = np.c_[bias(Xval),Xval]
Xtest = np.c_[bias(Xtest),Xtest]

def costFunciton(theta,X,y):
    m = X.shape[0]
    inner = X @ theta - y.flatten()
    a = inner.T @ inner
    return a/(2*m)

def costFunction_reg(theta,X,y,lamda):
    m = X.shape[0]
    _theta = theta[1:]
    reg_term = (lamda/(2*m))*(_theta.T @ _theta)
    return costFunciton(theta,X,y) + reg_term

def gradient(theta,X,y):
    m = X.shape[0]
    inner = X @ theta -y.flatten()
    grad = X.T @ inner
    return grad/m

def gradient_reg(theta,X,y,lamda):
    m = X.shape[0]
    reg_term = theta.copy()
    reg_term[0] = 0
    reg_term = (lamda/m) *reg_term
    return gradient(theta,X,y) + reg_term

theta = np.ones(X.shape[1])
grad = gradient_reg(theta,X,y,1)
print(grad)
cost = costFunction_reg(theta,X,y,1)
print(cost)

#================================================
#                线性回归拟合
#================================================
def LR(X,y,lamda):
  theta = np.zeros(X.shape[1])
  res = opt.minimize(fun = costFunction_reg,x0 = theta,args=(X,y,lamda),jac= gradient_reg)
  return res.x

finial_theta = LR(X,y,0)
plt.figure(figsize=(8,5))
plt.scatter(X[:,1:],y,c='r',marker = 'x')
plt.plot(X[:,1],X @ finial_theta)#X第一列是偏置
plt.grid(True)
plt.show()

#================================================
#                Bias and Variance
#================================================

def plot_learning_curve(X,y,Xval,yval,lamda):
  xx = range(1,X.shape[0]+1)
  train_cost = []
  validation_cost = []
  for i in xx:
    res = LR(X[:i],y[:i],lamda)
    tc_i = costFunction_reg(res,X[:i],y[:i],lamda)
    vc_i = costFunction_reg(res,Xval,yval,lamda)
    train_cost.append(tc_i)
    validation_cost.append(vc_i)

  plt.figure(figsize=(8,5))
  plt.plot(xx,train_cost,label = 'training cost')
  plt.plot(xx,validation_cost, label = 'cross validation cost')
  plt.legend()
  plt.xlabel('Number of training examples')
  plt.ylabel('Error')
  plt.title('Learning Curve for LR')
  plt.grid(True)
  plt.show()

plot_learning_curve(X,y,Xval,yval,0)
#该曲线是明显的欠拟合高偏差的曲线，因为两曲线非常接近，训练误差和交叉验证误差都很高


#================================================
#                多项式的线性回归
#================================================
# 建立多项式特征

def polyFeatrues(X,power):
  Xpoly = X.copy()
  for i in range(2,power+1):
    Xpoly = np.insert(Xpoly,Xpoly.shape[1],np.power(X[:,1],i),axis = 1)
  return Xpoly

Xpoly = polyFeatrues(X,6)

def get_ms(X):
  means = np.mean(X,axis = 0)#axis=0代表对列操作
  std = np.std(X,axis=0,ddof = 1)#ddof = 1代表求样本标准差
  return means,std

def normalizeFeature(X,means,std):

  X_norm = X.copy()
  X_norm[:,1:] = X_norm[:,1:]-means[1:]
  X_norm[:,1:] = X_norm[:,1:]/std[1:]
  return X_norm

tm,ts = get_ms(polyFeatrues(X,6))

X_norm = normalizeFeature(polyFeatrues(X,6),tm,ts)
Xval_norm = normalizeFeature(polyFeatrues(Xval,6),tm,ts)
Xtest_norm = normalizeFeature(polyFeatrues(Xtest,6),tm,ts)

def fittingCurve(means,std,lamda):
  theta = LR(X_norm,y,lamda)
  x =np.linspace(-75,55,50)
  xmat = x.reshape(-1,1)
  xmat = np.insert(xmat,0,1,axis=1)
  Xmat = polyFeatrues(xmat,6)
  Xmat_norm = normalizeFeature(Xmat,means,std)


  plt.plot(x,Xmat_norm @ theta)



plt.figure(figsize=(8,5))
plt.scatter(X[:,1:],y,c='r',marker = 'x')
plt.grid(True)
fittingCurve(tm,ts,0)
plt.show()

plot_learning_curve(X_norm,y,Xval_norm,yval,0)
#此时训练误差太小，但是验证误差很大，所以很明显是过拟合，高方差



#================================================
#                正则化参数的调整
#================================================

plt.figure(figsize=(8,5))
plt.scatter(X[:,1:],y,c='r',marker = 'x')
plt.grid(True)
fittingCurve(tm,ts,100)
plt.show()


plot_learning_curve(X_norm,y,Xval_norm,yval,100)
#此时惩罚太大，又欠拟合了

#利用肘部图来选择合适的lamda；
#这里用吴恩达老师常用的lamda参数来作为参考
l = [0.,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
cost_t = []
cost_cv = []
for lamda in l:
  theta = LR(X_norm,y,lamda)
  cost_t.append(costFunction_reg(theta,X_norm,y,0))
  cost_cv.append(costFunction_reg(theta,Xval_norm,yval,0))

plt.figure(figsize=(8,5))
plt.plot(l,cost_t,label='train')
plt.plot(l,cost_cv,label = 'cross validation')
plt.legend()
plt.grid(True)
plt.show()
#可以从图中看出，正则化参数为3时，cost最小
