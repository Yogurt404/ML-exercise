import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report#一个评价包

##===================================================================
#                       Visualized Data
##===================================================================

data = sio.loadmat('ex4data1.mat')

X_ = data['X']
y0 = data['y']

def polt_an_image(X):
    rand = np.random.randint(0,5000)
    image = X[rand]

    #reshape((20,20))是为了把图像还原成20*20像素的图像，transpose()是为了旋转图像
    fig,ax = plt.subplots(figsize=(1,1))
    ax.matshow(image.reshape((20,20)).transpose(),cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('it should be {}'.format(y[rand]))


def plot_100_image(X):
    rand = np.random.randint(0,5000,size=[100,1])
    image = X[rand]
    figure,ax_array = plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize=(8,8))
    for i in range(10):
        for j in range(10):
            ax_array[i,j].matshow(image[10*i+j].reshape((20,20)).transpose(),cmap = 'gray_r')
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

#plot_an_image(X_)
#plot_100_image(X_)

##=============================================================
#                       反向传播的代价函数
##=============================================================

X0 = np.ones(X_.shape[0]).reshape(-1,1)
X = np.c_[X0,X_]
print(X.shape)
y0 = y0.reshape(y0.shape[0])

y = []
for i in range(1,11):
    y.append((y0==i).astype(int))

y = np.array(y).transpose()
print(y)
print(y.shape)

Theta = sio.loadmat('ex4weights.mat')
theta1 = Theta['Theta1']
theta2 = Theta['Theta2']

print(theta1.shape,theta2.shape)
#扁平化theta
theta = np.concatenate((np.ravel(theta1),np.ravel(theta2)))# 扁平化theta1和theta2
print(theta.shape)
#定义sigmoid激活函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#定义前向反馈算法
def feed_forword(theta,X):
    theta1 = theta[:25*401].reshape(25,401)
    theta2 = theta[25*401:].reshape(10,26)
    a1= X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2_0 = np.ones(a2.shape[0])#插入一个偏置单元
    a2 = np.c_[a2_0,a2]
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    return a1,z2,a2,z3,a3

a1,z2,a2,z3,h = feed_forword(theta,X)
#print(h.shape)
#print(a1.shape,z2.shape,a2.shape)

#定义costfunction
def costFunction(theta,X,y):
    m = X.shape[0]
    a1,z2,a2,z3,h = feed_forword(theta,X)
    J = -y*np.log(h)-(1-y)*np.log(1-h)

    return J.sum()/m#sum是因为J变成了一个存放所有cost的矩阵 而不是单独的一个值

J = costFunction(theta,X,y)
print(J)

#定义正则化costfunction
def costFunction_reg(theta,X,y,lamda):
    m = X.shape[0]
    theta1 = theta[:25*401].reshape(25,401)
    theta2 = theta[25*401:].reshape(10,26)
    #正则化项不对第一项偏执项进行惩罚
    reg_term1 = (lamda/(2*m)) * np.power(theta1[:,1:],2).sum()
    reg_term2 = (lamda/(2*m)) * np.power(theta2[:,1:],2).sum()
    return costFunction(theta,X,y) + reg_term1 + reg_term2

r_J = costFunction_reg(theta,X,y,1)
print(r_J)

##=====================================================
#                     反向传播
##=====================================================

#先定义sigmoid函数的导数
def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def gradient(theta,X,y):
  m = X.shape[0]
  theta1 = theta[:25*401].reshape(25,401)
  theta2 = theta[25*401:].reshape(10,26)

  a1,z2,a2,z3,h = feed_forword(theta,X)
  d3 = h-y
  d2 = d3 @ theta2[:,1:] * sigmoid_gradient(z2)
  D2 = d3.T @ a2
  D1 = d2.T @ a1
  D = (1/len(X)) * np.concatenate((np.ravel(D1),np.ravel(D2)))
  return D
#def gradient(theta,X,y):
#    m = X.shape[0]
#    theta1 = theta[:25*401].reshape(25,401)
#    theta2 = theta[25*401:].reshape(10,26)
#    delta1 = np.zeros(theta1.shape)
#    delta2 = np.zeros(theta2.shape)

#    a1,z2,a2,z3,h = feed_forword(theta,X)

#对第i个单元进行操作
#    for i in range(m):
#        a1i = a1[i,:]#代表a1的第i个单元(1,401)
#        z2i = z2[i,:]#(1,25)
#        a2i = a2[i,:]#(1,26)在前向传播中加入了偏置单元

#        hi = h[i,:]#(1,10)
#        yi = y[i,:]
#        print(a1i.shape,z2i.shape,a2i.shape,hi.shape,yi.shape)


#        d3i = hi-yi#误差信号(1,10)

        #z2_0 = np.ones(1)
#        z2i = np.insert(z2i,0,np.ones(1))#插入一个偏置单元 z2i.shape变为(26,)
        #计算倒数第二层的i单元的误差
#        d2i = (theta2.T @ d3i) * sigmoid_gradient(z2i) 

        #更新误差信号
#        delta2 = delta2 + np.matrix(d3i).T @ np.matrix(a2i) #(1,10).T @(1,26)-> (10,26)
        #去除第一列的偏置单元
#        delta1 = delta1 + np.matrix(d2i[1:]).T @ np.matrix(a1i) # (1,25).T @ (1,401)->(25,401)

#        delta2 = delta2/m
#        delta1 = delta1/m

#        return delta1,delta2

#delta1, delta2 = gradient(theta,X,y)
#print(delta1.shape)#(25,401)
#print(delta2.shape)#(10,26)
#梯度检测
#在你的神经网络,你是最小化代价函数J(Θ)。
# 执行梯度检查你的参数,你可以想象展开参数Θ(1)Θ(2)成一个长向量θ。
# 通过这样做,你能使用以下梯度检查过程。

#def gradient_checking(theta,X,y,e):
#    def one_gradient(plus,minus):
#        return (costFunction_reg(plus,X,y,1) - costFunction_reg(minus,X,y,1))/(2*e)
#    theta_matrix = np.array(np.matrix(np.ones(theta.shape[0])).T @ np.matrix(theta))#展开成(10285*10285)
##    e_matrix = np.identity(len(theta)) * e
#    plus = theta_matrix + e_matrix
#    minus = theta_matrix - e_matrix
#    gradient_1 = []
#    for i in range(len(theta)):

#        grad_i = one_gradient(plus[i],minus[i])
#        gradient_1.append(grad_i)#

#    gradient_1 = np.array(gradient_1)
#    finial_gradient = gradient_reg(theta,X,y,1)
#    diff = np.linalg.norm(gradient_1-finial_gradient)/np.linalg.norm(gradient_1 + finial_gradient)
#    return diff
def gradient_checking(theta,X,y,e):
  def one_gradient(plus,minus):
    return (costFunction_reg(plus,X,y,1)-costFunction_reg(minus,X,y,1))/(2*e)
  gradient_1 = []
  for i in range(len(theta)):
    plus = theta.copy()
    minus = theta.copy()
    plus[i] = plus[i] + e
    minus[i] = minus[i] - e
    grad_i = one_gradient(plus,minus)
    gradient_1.append(grad_i)
  gradient_1 = np.array(gradient_1)
  final_theta = gradient_reg(theta,X,y,1)
  diff = np.linalg.norm(gradient_1 - final_theta)/np.linalg.norm(gradient_1 + final_theta)
  print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))


def gradient_reg(theta,X,y,lamda):
    m = X.shape[0]
    delta = gradient(theta,X,y)
    delta1 = delta[:25*401].reshape(25,401)
    delta2 = delta[25*401:].reshape(10,26)
    theta1 = theta[:25*401].reshape(25,401)
    theta2 = theta[25*401:].reshape(10,26)

    theta1[:,0] = 0
    reg_theta1 = (lamda/m) *theta1
    delta1 = delta1 + reg_theta1

    theta2[:,0] = 0
    reg_theta2 = (lamda/m) * theta2
    delta2 = delta2 + reg_theta2

    return np.concatenate((np.ravel(delta1),np.ravel(delta2)))#扁平化

diff = gradient_checking(theta,X,y,0.0001)
print(diff)

rand_theta = np.random.uniform(-0.12,0.12,10285)

res = opt.minimize(fun = costFunction_reg,x0 = rand_theta,args= (X,y,1),jac = gradient_reg)

print(res)

a1,z2,a2,z3,h = feed_forword(res.x,X)
y_pred = np.argmax(h,axis = 1) + 1
print(y_pred)
print(classification_report(y0,y_pred))

#===============================================
#                  隐藏层的可视化
#===============================================

# 理解神经网络是如何学习的一个很好的办法是，可视化隐藏层单元所捕获的内容。
# 通俗的说，给定一个的隐藏层单元，可视化它所计算的内容的方法是找到一个输入x，
# x可以激活这个单元
# 对于我们所训练的网络，注意到θ1中每一行都是一个401维的向量，代表每个隐藏层单元的参数。
# 如果我们忽略偏置项，我们就能得到400维的向量，这个向量代表每个样本输入到每个隐层单元的像素的权重。
# 因此可视化的一个方法是，reshape这个400维的向量为（20，20）的图像然后输出。
def plot_hidden_layer(theta):
    finial_theta1 = theta[:25*401]
    finial_theta1 = finial_theta1[:,1:]#去掉偏置单元得到一个25*400的权重序列

    fig,ax = plt.subplots(nrows = 5, ncols=5,sharey=True,sharex=True,figsize=(5,5))

    for r in range(5):
        for c in range(5):
            ax[r,c].matshow(finial_theta1[5*r+c].reshape((20,20)),cmap = matplotlib.cm.binary)

plot_hidden_layer(res.x)
plt.show()