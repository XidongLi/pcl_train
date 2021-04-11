# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct

import random
import math
from spectral import Spectral_Clustering_Graph,Unnormalized_Spectral_Clustering
import open3d as o3d
import random
import copy

from scipy import spatial
# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    #随机取点
    max_distance=1
    a=0
    p_number=0
    p_number_max=0
    p_number_stop=1500
    index=[]
    final_index=[] 
    while a==0:
        sample_list = [i for i in range(data.shape[0])]
        sample_list = random.sample(sample_list, 4) 
        random_data = data[sample_list,:]
    #平面方程
        p1=random_data[1,:]-random_data[0,:]
        p2=random_data[2,:]-random_data[0,:]
        n=np.cross(p1,p2)#法向量
        A=n[0]
        B=n[1]
        C=n[2]
        D=-(random_data[0,0]*n[0]+random_data[0,1]*n[1]+random_data[0,2]*n[2])
        #计算距离
        for i,row in enumerate(data):
            distance=abs(np.dot(n,row)+D)/math.sqrt(A*A+B*B+C*C)
            if distance<=max_distance:
                p_number=p_number+1
                index.append(i)
        if p_number>p_number_max :
            p_number_max=p_number
            final_index=index.copy()
        if p_number_max>=p_number_stop :
            a=1
        else :
            p_number=0
            index.clear()
    ground=data[final_index,:]
    segmengted_cloud=np.delete(data, final_index, 0)
    print('origin data points num:', data.shape[0])
    #return ground
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud,ground
    
# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
"""
def clustering(data):
    # 作业2
    # 屏蔽开始
    data=[data,0]
    data[1]=np.zeros((data[0].shape[0]),dtype=np.int16)
    data=tuple(data)
    W=Spectral_Clustering_Graph (data,10)
    clusters_index,k=Unnormalized_Spectral_Clustering(W,data)
    # 屏蔽结束
    return clusters_index,k
"""
def dbscan_clustering(data,r,r_min,ground):
    noise_cluster=np.zeros((2,3))
    L=[]
    
    other_point=o3d.geometry.PointCloud()
    class_=[0]*data.shape[0]
    visit=[0]*data.shape[0]
    
    RNN=spatial.KDTree(data)
    k=1
    
    a=0
    while a==0:
        for i in range(len(visit)) :
            if visit[i]==0:
                break
            else:
                pass
        if all(visit) or data.shape[0]==1:
            print('finish')
            break
        Cluster_C(RNN, data,i, r, r_min,k,visit,class_)
        r_color=np.array([random.random(),random.random(),random.random()])
        other_point.points = o3d.utility.Vector3dVector(data[np.where(np.asarray(class_) == k)])
        other_point.paint_uniform_color(r_color)
        noise_cluster=np.vstack((noise_cluster,data[np.where(np.asarray(class_) == -1)]))
        #print(data[np.where(np.asarray(class_) == -1)])
        data=np.delete(data,np.where(np.asarray(visit) == 1),0)
        #print(data.shape)
        RNN=spatial.KDTree(data)
        class_=[0]*data.shape[0]
        visit=[0]*data.shape[0]
        k=k+1
        L.append(copy.deepcopy(other_point)) 

    #print(noise_cluster)
    other_point.points=o3d.utility.Vector3dVector(np.delete(noise_cluster,np.array([0,1]),0))
    other_point.paint_uniform_color([1, 0, 0])
    L.append(copy.deepcopy(other_point))
    o3d.visualization.draw_geometries(L)
    L.append(ground)
    o3d.visualization.draw_geometries(L)# 定义点云的颜色
    #随机的点：data[i]
    
    #r:用于RNN搜索，r_min：dbscan
def Cluster_C(RNN,data,i,r,r_min,k,visit,class_):
    if visit[i]==1:
        return 0
    RNN_list=RNN.query_ball_point(data[i], r)
    if len(RNN_list)>=r_min :
        visit[i]=1
        for b in RNN_list:
            class_[b]=k
        for a in RNN_list:
            Cluster_C(RNN,data,a,r,r_min,k,visit,class_)
    else:
        visit[i]=1
        class_[i]=-1
        #print('noise')
        return 0
    
    
            
    

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
"""
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()
"""
def main():
    root_dir = 'C:/Users/Mers/Desktop/kitti_point_clouds/' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(origin_points)
       # o3d.visualization.draw_geometries([pcd])
        
        downpcd = pcd.voxel_down_sample(voxel_size=1.5)
        #o3d.visualization.draw_geometries([downpcd])
        segmented_points,ground_p = ground_segmentation(np.asarray(downpcd.points))
        
        ground= o3d.geometry.PointCloud() 
        ground.points = o3d.utility.Vector3dVector(ground_p)
        ground.paint_uniform_color([0, 0, 1])
        #o3d.visualization.draw_geometries([ground])
        dbscan_clustering(segmented_points,2,5,ground)
        
        """
        other=o3d.geometry.PointCloud() 
        other.points = o3d.utility.Vector3dVector(segmented_points)
        #o3d.visualization.draw_geometries([other])
        
        cluster_index ,k= clustering(segmented_points) 
        L=[]
        other_point=o3d.geometry.PointCloud()
        r_color=np.zeros((k,3))
        for i in range(k):
            r_color[i]=np.array([random.random(),random.random(),random.random()])
            print(r_color)
            other_point.points = o3d.utility.Vector3dVector(segmented_points[np.where(cluster_index == i)[0]])
            other_point.paint_uniform_color(r_color[i])
            L.append(copy.deepcopy(other_point))
        #L.append(ground)
        
        o3d.visualization.draw_geometries(L)# 定义点云的颜色
        """

        
if __name__ == '__main__':
    main()
