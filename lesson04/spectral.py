
import numpy as np
from sklearn import  datasets
import matplotlib.pyplot as plt
from scipy import spatial
from Kmean import K_Means

def Spectral_Clustering_Graph (data,k):#k为搜索邻域
    W=np.zeros(((data[0].shape)[0],(data[0].shape)[0]))
    root = spatial.KDTree(data[0], leafsize=10)
    for a,point in enumerate(data[0]):
        d,index=root.query(point, k) 
        for i in range(1,k):
            W[a,index[i]]=np.linalg.norm(point-data[0][index[i],:])
    return W
            
def Unnormalized_Spectral_Clustering(W,datas):
    D=np.diagflat(W.sum(axis=1))
    L=D-W
    eigenvalues,eigenvectors=np.linalg.eig(L)
    sort = np.argsort(-eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    A=[]
    for i in range(len(eigenvalues)-1):
       A.append(eigenvalues[i+1]-eigenvalues[i]) 
       if i>0 and A[i]>20*A[i-1] :
          k=i+1
          print(A)
          break
       elif i==(len(eigenvalues)-1):
           k=i
    V=eigenvectors[:,0:k]
    N=np.zeros((V.shape[0]),dtype=np.int16)
    data=(V,N)
    test=K_Means(data,k)
    a=1
    while a==1:
        for i,point in enumerate(data[0]):
            data[1][i]=test.predict(point)
        a=test.fit(data)
    return data[1],k
    #plt.scatter(datas[0][:,0],datas[0][:,1],c=data[1])
    #plt.show()
###生成数据 
""" 
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
###
#开始

W=Spectral_Clustering_Graph (noisy_circles,10)
Unnormalized_Spectral_Clustering(W,noisy_circles)

W=Spectral_Clustering_Graph (noisy_moons,10)
Unnormalized_Spectral_Clustering(W,noisy_moons)

W=Spectral_Clustering_Graph (blobs,10)
Unnormalized_Spectral_Clustering(W,blobs)

W=Spectral_Clustering_Graph (no_structure,10)
Unnormalized_Spectral_Clustering(W,no_structure)

W=Spectral_Clustering_Graph (aniso,10)
Unnormalized_Spectral_Clustering(W,aniso)

W=Spectral_Clustering_Graph (varied,10)
Unnormalized_Spectral_Clustering(W,varied)

"""


    