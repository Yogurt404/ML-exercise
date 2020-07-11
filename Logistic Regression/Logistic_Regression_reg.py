import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report#评价报告包 

##=============Visualize Data================
data = np.loadtxt('ex2data2.txt',delimiter=',')
print(data.shape,type(data))
pos_flag = data[:,2] == 1
neg_flag = data[:,2] == 0
pos = data[pos_flag]
neg = data[neg_flag]

#plt.scatter(pos[:,0],pos[:,1],marker='x',c='turquoise')
#plt.scatter(neg[:,0],neg[:,1],c = 'darkorange')
#plt.show()
##============Feature Mapping====================
#创建更复杂的特征
#():tuple []:list {}:diction
#字典里键一般是唯一的，如果重复最后的一个键值对会替换前面的，值不需要唯一

#从 Python 2.6开始，新增了一种格式化字符串的函数str.format()，基本语法是通过{}和:来代替以前的%。
# format函数支持通过位置、关键字、对象属性和下标等多种方式使用，
# 不仅参数可以不按顺序，也可以不用参数或者一个参数使用多次。
# 并且可以通过对要转换为字符串的对象的__format __方法进行扩展。

#f-string用大括号 {} 表示被替换字段，其中直接填入替换内容：

#def featureMapping(x1,x2,power):
#    data{"f{}{}".format(i-p,p):np.power(x1,i-p)*np.power(x2,p)
#    for i in np.arange(power+1)
#    for p in np.arange(i+1)
#    }#其中.format(i-p,p)代表了键值
#    return 

#非pandas
def featureMapping(x1,x2,power):
    out = np.zeros(x1.shape)

    for i in np.arange(power+1):
        for p in np.arange(i+1):
            out = np.hstack((out,np.power(x1,i-p)*np.power(x2,p)))

    return out[:,1:]
X_1 = data[:,0].reshape(-1,1)
X_2 = data[:,1].reshape(-1,1)
y = data[:,2]

X = featureMapping(X_1,X_2,6)
print(X)
print(X.shape,type(X))
print(y.shape,type(y))
#===============Feature Normalization=======================
#X = (X - X.mean())/X.std()
#y = (y - y.mean())/y.std()
#===============Regularzed Cost Function====================
theta = np.zeros(X.shape[1])
print(theta.shape,type(theta))
#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#逻辑回归的代价函数
def costFunction(theta,X,y):
    z = X @ theta
    h = sigmoid(z)

    J = np.mean(-y*np.log(h)-(1-y)*np.log(1-h))
    return J

def costFunction_reg(theta,X,y,lamda):
    _theta = theta[1:]
    m = X.shape[0]
    reg_term = (lamda/(2*m)) * np.power(_theta,2).sum()
    return costFunction(theta,X,y)+reg_term

J = costFunction_reg(theta,X,y,1)
print(J)

##===============Gradient Descent==================
def gradient(theta,X,y):
    z = X @ theta
    h = sigmoid(z)
    m = X.shape[0]

    gradient = X.T @ (h-y)
    return gradient/m


def gradient_reg(theta,X,y,lamda):
    _theta = theta[1:]
    m = X.shape[0]
    reg_theta = (lamda/m)*_theta
    theta0 = np.array([0])
    reg_term = np.r_[theta0,reg_theta]
    return gradient(theta,X,y) + reg_term

g = gradient_reg(theta,X,y,1)

#**theta必须为第一个参数且其shape必须为(n,)**即一维数组
res = opt.minimize(fun = costFunction_reg,x0 = theta,args=(X,y,1),jac=gradient_reg)
print(res.x)
##===================Prediction and Accuracy==================
def predict(theta,X):
    probability = sigmoid(X @ theta)
    return [1 if x>= 0.5 else 0 for x in probability]

predictions = predict(res.x,X)
correct = [1 if a==b else 0 for (a,b) in zip(predictions,y)]
accuracy = sum(correct)/len(X)
print(accuracy)

##===================Decision Boundary=======================
#[X,Y] = meshgrid(x) 与 [X,Y] = meshgrid(x,x) 相同，
# 并返回网格大小为 length(x)×length(x) 的方形网格坐标。
x = np.linspace(-1,1.5,250)
[xx,yy] = np.meshgrid(x,x)


z = featureMapping(xx.ravel().reshape(-1,1),yy.ravel().reshape(-1,1),6)
z = z @ res.x
z = z.reshape(xx.shape)


plt.scatter(pos[:,0],pos[:,1],marker='x',c='turquoise')
plt.scatter(neg[:,0],neg[:,1],c = 'darkorange')
plt.contour(xx,yy,z,0)
plt.show()


