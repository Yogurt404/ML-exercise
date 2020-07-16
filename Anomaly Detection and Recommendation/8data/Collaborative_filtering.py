import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt

##========================================================
#                       可视化数据
##========================================================
data = sio.loadmat('8data/ex8_movies.mat')

#矩阵Y(nm, nu)是不同电影的不同用户的评分，行数为电影数目，列数为用户数目。

#矩阵R是二进制指示矩阵，R(i, j)=1 表示用户j 对电影i 有评分，R(i, j)=0 表示用户j对电影i没有评分。

#协同过滤的目标是预测用户还没有评分的电影的评分，也就是R(i, j) = 0的项。
# 这将允许我们向用户推荐预测收视率最高的电影。

Y, R = data['Y'],data['R']

print(Y.shape,R.shape)
# Y:1682个电影和943位用户
nm = Y.shape[0]
nu = Y.shape[1]

plt.figure(figsize=(5,5*(1682./943.)))
plt.imshow(Y,cmap='rainbow')
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=10)
plt.xlabel('Users (%d)'%nu,fontsize=10)
#plt.show()

##========================================================
#                       协同过滤
##========================================================
data2= sio.loadmat('8data/ex8_movieParams.mat')
#1682个电影，943位用户，10个特征
X = data2['X']#1682个电影包含的10种特征
Theta = data2['Theta']#943位用户对这10种特征的偏好程度
nm = int(data2['num_movies'])
nu = int(data2['num_users'])
nf = int(data2['num_features'])


#代价函数
def expand(X,theta):
  return np.r_[X.flatten(),theta.flatten()]


def extract(exp,nu,nm,nf):
  return exp[:nm*nf].reshape(nm,nf),exp[nm*nf:].reshape(nu,nf)

def costFunction(param,Y,R,nu,nm,nf,lamda):
    X,theta = extract(param,nu,nm,nf)

    # (X@Theta)*R含义如下： 因为X@Theta是我们用自定义参数算的评分，但是有些电影本来是没有人
    # 评分的，存储在R中，0-1表示。将这两个相乘，得到的值就是我们要的已经被评分过的电影的预测分数。
    error = 0.5 * np.square((X @ theta.T - Y)*R).sum()
    reg1 = (lamda/2)*np.square(theta).sum()
    reg2 = (lamda/2)*np.square(X).sum()

    return error + reg1+reg2

def gradient(param,Y,R,nu,nm,nf,lamda):
    X,theta = extract(param,nu,nm,nf)

    X_grad = ((X @ theta.T-Y)*R)@theta + lamda*X
    theta_grad = ((X@theta.T-Y)*R).T @ X + lamda*theta
    return expand(X_grad,theta_grad)

def checkGradient(param,Y,R,nu,nm,nf,lamda):
    grad = gradient(param,Y,R,nu,nm,nf,lamda)

    e = 0.0001
    e_vec = np.zeros(len(param))

    for i in range(10):
        idx = np.random.randint(0,len(param))
        e_vec[idx] = e
        loss1 = costFunction(param-e_vec,Y,R,nu,nm,nf,lamda)
        loss2 = costFunction(param+e_vec,Y,R,nu,nm,nf,lamda)
        a = (loss2-loss1)/2*e

        e_vec[idx] = 0
        diff = np.linalg.norm(a-grad[idx])/np.linalg.norm(a+grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(a, grad[idx], diff))


checkGradient(expand(X,Theta),Y,R,nu,nm,nf,0)
checkGradient(expand(X,Theta),Y,R,nu,nm,nf,1.5)

##========================================================
#                       训练模型
##========================================================
movie_idx = {}
f = open('8data/movie_ids.txt',encoding= 'gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
#输入用户评分
my_ratings = np.zeros((1682,1))

my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(len(my_ratings)):
  if my_ratings[i]>0:
    print(my_ratings[i],movie_idx[i])

#把新用户的数据添加到原来的Y，R中
Y = np.c_[Y,my_ratings]
R = np.c_[R,my_ratings!=0]
nm,nu = Y.shape

X= np.random.random((nm,nf))
Theta = np.random.random((nu,nf))

param = expand(X,Theta)
lamda = 10

res = opt.minimize(fun=costFunction,x0=param,args=(Y,R,nu,nm,nf,lamda),jac = gradient, method='TNC',options={'maxiter': 100})
final_param = res.x
fit_X,fit_theta = extract(final_param,nu,nm,nf)
#所有用户
pred = fit_X @ fit_theta.T
#取最后一个用户
pred = pred[:,-1]
#排序
pred_sort = np.argsort(pred)[::-1]

print("Top recommendations for you:")
for i in range(10):
    print('Predicting rating %0.1f for movie %s.' \
          %(pred[pred_sort[i]],movie_idx[pred_sort[i]]))

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.'% (my_ratings[i],movie_idx[i]))