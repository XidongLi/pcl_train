import numpy as np
from sklearn import  datasets
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal


def Multi_Gaussian(point,mean,covariance):
        var = multivariate_normal(mean, covariance)
        return var.pdf(point)
    
class GMM(object):
    def __init__(self,data,n_clusters, max_iter=50,tolerance=0.001):
        self.k_ = n_clusters
        self.max_iter_ = max_iter
        self.tolerance_ = tolerance
        self.mean=np.zeros((self.k_, 2))#中心点
        self.covariance=np.zeros((self.k_,2,2))
        self.weight=np.ones(self.k_)
        self.avr_id=np.arange(self.k_)#类标签
        for i in range(self.k_):
            self.mean[i,:]=data[0][random.randint(0, 1499),:]
            self.covariance[i,:,:]=np.array([[1,0],[0,1]])
            #self.weight[i]=random.random()
        self.weight=self.weight/self.weight.sum()
        #self.mean=np.array([[0.5,0.],[0.2,0.8],[0.85,0.8]])
        self.gamma=np.zeros(((data[0].shape)[0],self.k_))
        
    def predict(self, data,index):#传入单个点及其index
        p=np.zeros(self.k_+1)
        i=-1
        for a in range(self.k_):
            p[a+1]=self.weight[a]*Multi_Gaussian(data,self.mean[a,:],self.covariance[a,:,:])
            if p[a+1]>p[0] :
                p[0]=p[a+1]
                i=a
        p[0]=0
        for a in range(self.k_):
            self.gamma[index,a]=p[a+1]/p.sum()
        
        return i#返回类标签
    
    def fit(self, p_datas):#传入整个数据
        
        old_mean=np.copy(self.mean)
        N=self.gamma.sum(axis=0)
        matrix=np.zeros((2,2))
        for i in range(self.k_):
            self.mean[i,:]=(np.vstack((self.gamma[:,i],self.gamma[:,i])).T*p_datas[0]).sum(axis=0)/N[i]
            self.weight[i]=N[i]/(p_datas[0].shape)[0]
            for index,point in enumerate(p_datas[0]):
                matrix=matrix+(point-self.mean[i,:]).reshape(1,2).T@(point-self.mean[i,:]).reshape(1,2)*self.gamma[index,i]
            self.covariance[i,:,:]=np.copy(matrix/N[i])
            matrix=np.zeros((2,2))
        self.max_iter_=self.max_iter_-1
        self.gamma=np.zeros(((p_datas[0].shape)[0],self.k_))#刷新数据后清零
        """
        if np.linalg.norm(self.mean-old_mean) <=self.tolerance_:
            print("收敛，迭代次数为",300-self.max_iter_)
            return 0
        """
        if self.max_iter_==0:
            return 0
        else :
            return 1


# 模拟原始数据
np.random.seed(0)
print('start generate datasets ...')
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
no_structure=list(no_structure)
no_structure[1]=np.zeros((1500),dtype=np.int16)
no_structure=tuple(no_structure)
print('datasets generated over')
# ============
"""
plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1],c=noisy_circles[1])
plt.show()

test=GMM(noisy_circles,n_clusters=2)
a=1
while a==1:
    for i,point in enumerate(noisy_circles[0]):
        noisy_circles[1][i]=test.predict(point,i)
    a=test.fit(noisy_circles)
plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1],c=noisy_circles[1])
plt.show()

"""





"""
plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1],c=noisy_moons[1])
plt.show()

test=GMM(noisy_moons,n_clusters=2)
a=1
while a==1:
    for i,point in enumerate(noisy_moons[0]):
        noisy_moons[1][i]=test.predict(point,i)
    a=test.fit(noisy_moons)

    plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1],c=noisy_moons[1])
    plt.show()
"""
    
"""
plt.scatter(blobs[0][:,0],blobs[0][:,1],c=blobs[1])
plt.show()

test=GMM(blobs,n_clusters=3)
a=1
while a==1:
    for i,point in enumerate(blobs[0]):
        blobs[1][i]=test.predict(point,i)
    a=test.fit(blobs)
plt.scatter(blobs[0][:,0],blobs[0][:,1],c=blobs[1])
plt.show()
 """

plt.scatter(no_structure[0][:,0],no_structure[0][:,1],c=no_structure[1])
plt.show()

test=GMM(no_structure,n_clusters=3)
a=1
while a==1:
    for i,point in enumerate(no_structure[0]):
        no_structure[1][i]=test.predict(point,i)
    a=test.fit(no_structure)
    plt.scatter(no_structure[0][:,0],no_structure[0][:,1],c=no_structure[1])
    plt.show()


"""
plt.scatter(aniso[0][:,0],aniso[0][:,1],c=aniso[1])
plt.show()

test=GMM(aniso,n_clusters=3)
a=1
while a==1:
    for i,point in enumerate(aniso[0]):
        aniso[1][i]=test.predict(point,i)
    a=test.fit(aniso)
plt.scatter(aniso[0][:,0],aniso[0][:,1],c=aniso[1])
plt.show()


plt.scatter(varied[0][:,0],varied[0][:,1],c=varied[1])
plt.show()

test=GMM(varied,n_clusters=3)
a=1
while a==1:
    for i,point in enumerate(varied[0]):
        varied[1][i]=test.predict(point,i)
    a=test.fit(varied)
plt.scatter(varied[0][:,0],varied[0][:,1],c=varied[1])
plt.show()
"""
