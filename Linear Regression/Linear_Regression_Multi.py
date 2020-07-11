import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = np.loadtxt('ex1data2.txt', delimiter = ',')
print(data.shape)


X_1 = data[:,0].reshape(-1,1)
X_2 = data[:,1].reshape(-1,1)
Y_1 = data[:,-1].reshape(-1,1)
X1_data = (X_1 - X_1.mean())/X_1.std()
X2_data  = (X_2 - X_2.mean())/X_2.std()
Y_data  = (Y_1 - Y_1.mean())/Y_1.std()

m = len(X_1)
X_0 = np.ones(m).reshape(-1,1)
X = np.c_[X_0,X1_data,X2_data]
y = Y_data

theta = np.zeros((3,1))
iterations = 1000
alpha = 0.01

def costFunction(X,y,theta):
    m = X.shape[0]
    error = X@theta - y

    J = error.T @ error
    return J/(2*m)

def gradientDescent(X,y,theta,alpha,iterations):
    m = X.shape[0]
    J_list = [costFunction(X,y,theta)]
    _theta = theta


    for _ in range(iterations):
        error = X @ _theta - y
        gradient = X.T @ error
        _theta = _theta - alpha * gradient / m

        J_list.append(costFunction(X,y,_theta))
    return _theta,J_list

_theta,J_list = gradientDescent(X,y,theta,alpha,iterations)

J_list = np.array(J_list).reshape(-1,1)
print(_theta)
print(J_list.shape)

plt.plot(np.arange(0,J_list.size,1),J_list,'-',)
plt.show()
