import numpy as np
from sklearn import  datasets
import matplotlib.pyplot as plt
import random

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self ,data,n_clusters=2, tolerance=0.00000001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.avr=np.zeros((n_clusters, data[0].shape[1]))#中心点
        self.avr_id=np.arange(n_clusters)#类标签
        for i in range(n_clusters):
            self.avr[i,:]=data[0][random.randint(0, 1499),:]
    def predict(self, data):#传入单个点
        dist=10000
        i=-1
        for a,u in enumerate(self.avr):
            dist1 = np.linalg.norm(u - data) 
            if dist1<dist :
                dist=dist1
                i=a
        return i#返回类标签
        # 屏蔽结束

    def fit(self, p_datas):#传入整个数据
        old_avr=np.copy(self.avr)
        result =[]
        for i in range(self.k_):
            result.append(self.avr[i,:])
        for i,point in enumerate(p_datas[0]):
            result[p_datas[1][i]]=np.vstack((result[p_datas[1][i]],point))
        for i in range(self.k_):
            self.avr[i,:]=np.mean(result[i], axis=0)
        self.max_iter_=self.max_iter_-1
        
        if np.linalg.norm(self.avr-old_avr) <=self.tolerance_:
            print("收敛，迭代次数为",300-self.max_iter_)
            return 0
    
        if self.max_iter_==0:
            return 0
        else :
            return 1
        # 屏蔽结束
# ============
# 模拟原始数据
# ============
def main():
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


    plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1],c=noisy_circles[1])
    plt.show()
    test=K_Means(noisy_circles,2)
    a=1
    while a==1:
        for i,point in enumerate(noisy_circles[0]):
            noisy_circles[1][i]=test.predict(point)
        a=test.fit(noisy_circles)

    plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1],c=noisy_circles[1])
    plt.show()

    plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1],c=noisy_moons[1])
    plt.show()

    test=K_Means(noisy_moons,n_clusters=2)
    a=1
    while a==1:
        for i,point in enumerate(noisy_moons[0]):
            noisy_moons[1][i]=test.predict(point)
        a=test.fit(noisy_moons)

    plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1],c=noisy_moons[1])
    plt.show()

    plt.scatter(blobs[0][:,0],blobs[0][:,1],c=blobs[1])
    plt.show()

    test=K_Means(blobs,n_clusters=3)
    a=1
    while a==1:
        for i,point in enumerate(blobs[0]):
            blobs[1][i]=test.predict(point)
        a=test.fit(blobs)
    plt.scatter(blobs[0][:,0],blobs[0][:,1],c=blobs[1])
    plt.show()

    plt.scatter(no_structure[0][:,0],no_structure[0][:,1],c=no_structure[1])
    plt.show()

    test=K_Means(no_structure,n_clusters=3)
    a=1
    while a==1:
        for i,point in enumerate(no_structure[0]):
            no_structure[1][i]=test.predict(point)
        a=test.fit(no_structure)
    plt.scatter(no_structure[0][:,0],no_structure[0][:,1],c=no_structure[1])
    plt.show()

    plt.scatter(aniso[0][:,0],aniso[0][:,1],c=aniso[1])
    plt.show()

    test=K_Means(aniso,n_clusters=3)
    a=1
    while a==1:
        for i,point in enumerate(aniso[0]):
            aniso[1][i]=test.predict(point)
        a=test.fit(aniso)
    plt.scatter(aniso[0][:,0],aniso[0][:,1],c=aniso[1])
    plt.show()

    plt.scatter(varied[0][:,0],varied[0][:,1],c=varied[1])
    plt.show()

    test=K_Means(varied,n_clusters=3)
    a=1
    while a==1:
        for i,point in enumerate(varied[0]):
            varied[1][i]=test.predict(point)
        a=test.fit(varied)
    plt.scatter(varied[0][:,0],varied[0][:,1],c=varied[1])
    plt.show()

if __name__ == '__main__':
    main()



