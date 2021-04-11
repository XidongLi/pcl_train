# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import math
import numpy as np


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    h_set = []
    # 作业3
    # 屏蔽开始
    N=point_cloud
    #提取每列最小值
    N_min=N.min(axis=0)
    print(N_min)
    #提取每列最大值
    N_max=N.max(axis=0)
    print(N_max)
    #计算x方向上的格子数
    D_x=math.ceil((N_max[0]-N_min[0])/leaf_size)
    print(D_x)    
    #计算y方向上的格子数
    D_y=math.ceil((N_max[1]-N_min[1])/leaf_size)
    print(D_y)
    #计算z方向上的格子数
    D_z=math.ceil((N_max[2]-N_min[2])/leaf_size)
    print(D_z)
    for row in N:
        h_x=round((row[0]-N_min[0])/leaf_size)
        h_y=round((row[1]-N_min[1])/leaf_size)
        h_z=round((row[2]-N_min[2])/leaf_size)
        h=h_x+h_y*D_x+h_z*D_x*D_y
        h_set=np.hstack((h_set,h))#h_set为所有点的h的集合
    #对h进行排序
    sort = h_set.argsort()[::-1]
    h_set = h_set[sort]
    N = N[ sort,:]
    a=h_set[0]
    X=N[0,0:3]
    filtered_points=N[0,:]
    for i in range(0,h_set.size):
        if a==h_set[i] :#若h值相同，则为同一区域
            X=np.vstack((X,N[i,:]))
        else:#h值不相同
            if len(X.shape)==1:#区域内只有一个点，无法取平均，直接输出
                filtered_points =np.vstack((filtered_points,X))
            else:
                X_avr=np.mean(X, axis=0)#区域内有多个点，取平均并输出
                filtered_points =np.vstack((filtered_points,X_avr))
            a=h_set[i]#重置h，以便用来比较
            X=N[i,:]#重置X
           
            
    # 屏蔽结束
    print(filtered_points)
    # 把点云格式改成array，并对外返回
    #filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def numpy_read_txt(pc_txt_path):
  data = np.genfromtxt(pc_txt_path,delimiter=",")
  pc = data[:,:3]
  return pc

def main():
    pc_txt_path = "/Users/Mers/Desktop/pc/airplane_0542.txt"
    pc = numpy_read_txt(pc_txt_path)
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc)
    print('可视化原始点云')
    o3d.visualization.draw_geometries([pcd])
    pcd.points=o3d.utility.Vector3dVector(voxel_filter(pcd.points, 0.1))
    print('可视化降采样后的点云')
    o3d.visualization.draw_geometries([pcd])
    
   # N=np.asarray(pcd.points)
   # print(N.max(axis=0))
    """
    # 加载自己的点云文件
    file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
"""
if __name__ == '__main__':
    main()
