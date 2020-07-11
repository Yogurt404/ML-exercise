import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report#评价报告包 
data = np.loadtxt('ex2data1.txt',delimiter=',')

##===============Visualize Data=========================
#分离正负样本，argwhere得到一个索引值的列表,用where会得到一个numpy-like的tuple 无法运算

#pos_index = np.where(data[:,2]==1)
#neg = np.argwhere(data[:,2]==0)
pos_flag = data[:,2] == 1
neg_flag = data[:,2] == 0
pos = data[pos_flag]
neg = data[neg_flag]




#plt.scatter(pos[:,0],pos[:,1],marker='x',c='turquoise')
#plt.scatter(neg[:,0],neg[:,1],c = 'darkorange')
#plt.show()

#====================特征提取=====================
X_1 = data[:,0].reshape(-1,1)
X_2 = data[:,1].reshape(-1,1)
y = data[:,2]

X_0 = np.ones(len(X_1)).reshape(-1,1)

X = np.c_[X_0,X_1,X_2]
print(X)



def sigmoid(z):
    return 1/(1+np.exp(-z))

#fig,ax = plt.subplots(figsize=(8,6))
#ax.plot(np.arange(-10,10,step=0.01),sigmoid(np.arange(-10,10,step=0.01)))
#plt.show()
#theta = np.random.rand(3,1)
theta = np.zeros(3)#(2,)表示一位数组有两个元素 (2,1)表示一个二维数组，每行有一个元素
# z= X @ theta
print(X.shape)
#print(y.shape)
#print(theta.shape)
def costFunction(theta,X,y):
    #m = y.size
    z = X @ theta
    h = sigmoid(z)

    #J = -1 * ((1/m) *np.log(h).T @ y +np.log(1-h).T @ (1-y))
    J= np.mean(-y * np.log(h)-(1-y)*np.log(1-h))

    return J

J = costFunction(theta,X,y)
#print(J)
def gradient(theta,X,y):
    m = y.shape[0]
    z = X @ theta
    h = sigmoid(z)
    

    grad = X.T @ (h-y) # (3,100) @ ()
    return grad/m

grad = gradient(theta,X,y) #得到一个(1,3)的矩阵
print(grad)
reg = opt.minimize(fun = costFunction,x0=theta,args=(X,y),jac = gradient)
print(reg)
final_theta = reg.x


##===============Predictions and Accuarcy=================
#X @ theta = 0 决策边界
def predict(theta,X):
    probability = sigmoid(X @ theta)
    return [1 if x>= 0.5 else 0 for x in probability]

predictions = predict(final_theta,X)
correct = [1 if a==b else 0 for (a,b) in zip(predictions,y)]
accuracy = sum(correct)/len(X)
print(accuracy)

##==============Decision Boundary=====================
x1 = np.arange(0,130,step = 0.1)
x2 = -1*(final_theta[0]+x1*final_theta[1])/final_theta[2]

plt.scatter(pos[:,0],pos[:,1],marker='x',c='turquoise')
plt.scatter(neg[:,0],neg[:,1],c = 'darkorange')
plt.plot(x1,x2)
plt.show()