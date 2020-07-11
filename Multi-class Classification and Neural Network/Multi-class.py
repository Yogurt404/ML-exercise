import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report#一个评价包
##=================Visualize Data=====================
#该训练样本为5,000张20*20的书写数字的灰度图。
#data = sio.loadmat('ex3data1.mat')

#X = data['X']
#X = X.transpose
#y = data['y']
#y = y.reshape(y.shape[0])

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y

X, y =load_data('ex3data1.mat') 
#X的shape是(5000,400)
#每一行都代表一张图像，每张图像的20*20灰度值被展开成一列
#print(X.shape)
#print(y.shape)

#画出一个图像
def plot_an_image(X):
    rand = np.random.randint(0,5000)
    image = X[rand,:]
    fig,ax = plt.subplots(figsize=(1,1))
    ax.matshow(image.reshape((20,20)),cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('it should be {}'.format(y[rand]))

#plot_an_image(X)

def plot_100_image(X):
    index = np.random.randint(0,5000,size=[100,1])
    image = X[index,:]
    fig,ax_array = plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize=(8,8))

    for row in range(10):
        for column in range(10):
            ax_array[row,column].matshow(image[10*row+column].reshape((20,20)),cmap = 'gray_r')

    plt.xticks([])
    plt.yticks([])
    plt.show()
    

plot_100_image(X)

##==============Cost Function and Gradient===================
#一维模型的梯度下降
#===========Initializaiton==========
X0 = np.ones(X.shape[0]).reshape(-1,1)
X_data = np.c_[X0,X]

y_m = []


for i in range(1,11):
    y_m.append((y == i).astype(int))


y_m = [y_m[-1]]+y_m[:-1] # 把最后一行放到第一行
y_m = np.array(y_m)
print(y)
print(y_m.shape)
print(y_m,y_m[0].shape)

theta = np.zeros(X_data.shape[1])
#==============================================================
#sigmoid激活函数，梯度和代价函数的定义
#==============================================================
def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(theta,X,y):
    z = X @ theta
    h = sigmoid(z)
    J = np.mean(-y*np.log(h)-(1-y)*np.log(1-h))
    return J

def costFunction_reg(theta,X,y,lamda):
    _theta = theta[1:]
    m = X.shape[0]
    reg_term = lamda/(2*m)*np.power(_theta,2).sum()
    
    return costFunction(theta,X,y) + reg_term


def gradient(theta,X,y):
    z = X @ theta
    h = sigmoid(z)
    m = X.shape[0]

    grad = X.T @ (h-y)
    return grad/m

def gradient_reg(theta,X,y,lamda):
    _theta = theta[1:]
    m = X.shape[0]
    reg_theta = (lamda/m) * _theta
    theta0 = np.array([0])
    reg_term = np.r_[theta0,reg_theta]

    return gradient(theta,X,y) + reg_term

#grad = gradient_reg(theta,X_data,y_m[0],1)
#print(grad)

#==============================================================
#取一列y[0]并且做出预测，计算准确度
#==============================================================
J = costFunction_reg(theta,X_data,y_m[0],1)
print(J)

res = opt.minimize(fun = costFunction_reg,x0 = theta,args=(X_data,y_m[0],1),jac=gradient_reg)
#print(res.x)


def predict(theta,X):
    probability = sigmoid(X @ theta) 
    return (probability >= 0.5).astype(int)

print(X_data)
print(sigmoid(X_data @ res.x))
print(res.x.shape)
prediction = predict(res.x,X_data)
print(prediction)
accuracy = np.mean(y_m[0]==prediction)
print(accuracy)



#==============================================================
#==============================================================
#                       K 维模型的训练
#==============================================================
#==============================================================

#取k个final theta
k_theta = []

for k in range(10):
    res = opt.minimize(fun = costFunction_reg, x0 = theta, args=(X_data,y_m[k],1),jac = gradient_reg)
    k_theta.append(res.x)

k_theta = np.array(k_theta)
print(k_theta.shape)

#think about the shape of k_theta, now you are making X * theta.T
# (5000,401) * (401,10).T = (5000,10)
prob_matrix = sigmoid(X_data @ k_theta.T)
np.set_printoptions(suppress=True)
print(prob_matrix)

#np.argmax 用来输出最大值的索引.
y_pred = np.argmax(prob_matrix,axis=1)
print(y_pred,y_pred.shape)

#y_pred 得出的是所有图片对所有类别相似的概率，现在取出其中概率最大的
#即最相似的数字

y1 = y.copy()
y1[y1==10]=0

print(classification_report(y1,y_pred))
#上面使用了多类logistic回归，然而logistic回归不能形成更复杂的假设，因为它只是一个线性分类器。

#接下来我们用神经网络来尝试下，神经网络可以实现非常复杂的非线性的模型。我们将利用已经训练好了的权重进行预测。


#==============================================================
#==============================================================
#                       神经网络以及前向反馈
#==============================================================
#==============================================================

weight = sio.loadmat('ex3weights.mat')
theta1 = weight['Theta1']
theta2 = weight['Theta2']

print(theta1.shape)
print(theta2.shape)

data = sio.loadmat('ex3data1.mat')

X1 = data['X']
y1 = data['y']
y1 = y1.reshape(y.shape[0])
X1 = np.c_[X0,X1]
print(X1.shape)
print(y1.shape)

#=========前馈预测============
a1 = X1 # (5000,401)

z2 = a1 @ theta1.T # (5000,401) * (401,25) = (5000,25)

z2_0 = np.ones(z2.shape[0])
z2 = np.c_[z2_0,z2]

print(z2.shape)

a2 = sigmoid(z2)

z3 = a2 @ theta2.T#(5000,26) * (26,10)

a3 = sigmoid(z3)
print(a3)

y1_pred = np.argmax(a3,axis = 1) +1 # numpy is 0 base index, +1 for matlab convention

print(y1_pred.shape)

print(classification_report(y1,y1_pred))