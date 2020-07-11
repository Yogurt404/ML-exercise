import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set(context="notebook", style = "whitegrid",palette = "dark")
##===============Part 2: Plotting==================
data = np.loadtxt('ex1data1.txt', delimiter = ',')


X_1 = data[:,:-1]
Y_1 = data[:,-1]
m = len(Y_1)
print(m)
#plt.scatter(X_data,Y_data)
#plt.grid()
#plt.show()

# 特征缩放
X_data  = (X_1 - X_1.mean())/X_1.std()
Y_data  = (Y_1 - Y_1.mean())/Y_1.std()

##================Part 3: Gradient Descent=================

X_0 = np.ones(m)
X = np.c_[X_0,X_data]
y = Y_data.reshape((-1,1))# reshape成一列 行数自动

print(X.shape)
print(y.shape)

#theta = np.random.rand(2,1) # 省略了一步 应该是 theta =  (2,1) , np.zeros(theta)
#print(theta)
theta = np.zeros((2,1))
print(theta.shape)
iterations =1000
alpha = 0.001


def costFunction(X,y,theta):
    m = X.shape[0]

    error = X @ theta - y

    J = (error.T @ error)/(2*m)
    return J

J =costFunction(X,y,theta)
print(J)

def gradientDescent(X,y,theta,alpha,iterations):
    m = X.shape[0]
    J_list = [costFunction(X,y,theta)]
    theta_1 = theta
    theta_list = [0,0]

    for _ in range(iterations):
        error = X @ theta - y
        gradient = X.T @ error
        theta_1 = theta_1 - alpha *gradient /m
        theta_list.append(theta_1)


        J_list.append(costFunction(X,y,theta_1))
        if J_list[_] < 0.005:
            break
        



       # print('Round %s'%(i))
    J_list = np.array(J_list)
    return theta_1,J_list,theta_list

theta_1,J_list,theta_list = gradientDescent(X,y,theta,alpha,iterations)

#把列表转换为ndarray
J_list = J_list[:,0,0]
J_list = J_list.reshape(-1,1)
theta_list = np.array(theta_list)
theta_list = theta_list.reshape(-1,2)
print(theta_list.shape)
#print(J_list)
print(theta_1[0])
print(theta_1[1])

#plt.plot(X[:,-1],X @ theta_1,'-',)
#plt.scatter(X_data,Y_data,marker='x')
#plt.show()

#===============Visualize J(tehta)=========================#
theta0 = np.linspace(-1,1,100)
theta1 = np.linspace(-1,1,100)
J_vals = np.zeros((len(theta0),len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = np.array([[theta0[i],theta1[j]]]).reshape(-1,1)
        J_vals[i,j]= costFunction(X,y,t)

theta0,theta1 = np.meshgrid(theta0,theta1)


fig = plt.figure()
ax = fig.gca(projection= '3d')
ax.plot_surface(theta0,theta1,J_vals)
plt.show()

plt.figure()
plt.contour(theta0,theta1,J_vals)
plt.plot(theta_1[0],theta_1[1],'rx')
plt.show()


 