import numpy as np
import time
import struct
import scipy.spatial 
import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet
import open3d as o3d

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

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8 #8NN
    radius = 1

    root_dir = '/Users/Mers/Desktop/WS2021/三维点云处理/lesson02/000000.bin' # 数据路径
    db_np = read_velodyne_bin(root_dir) #原始点云数据
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(db_np)
    o3d.visualization.draw_geometries([pcd])
    """
    print("Numpybrute-force search")
    force_min_distance=np.array([1e10])
    begin_t = time.time()
    row=db_np[0,:]
    for row_2 in db_np:
        force_min_distance=np.vstack((force_min_distance,np.array(np.linalg.norm(row-row_2))))
    sort=np.argsort(force_min_distance)[1:k+1]
    construction_time_sum = time.time() - begin_t
    print("Numpybrute-force search查询一个点所需时间：",construction_time_sum)
    
    
    print('scipy.spatial.KDTree')
    begin_t = time.time()
    scipy_kdtree=scipy.spatial.KDTree(db_np,leaf_size)
    construction_time_sum = time.time() - begin_t
    print("建立scipy的kdtree所需时间：",construction_time_sum)
    begin_t = time.time()
    for i in range(10000):
        row=db_np[i,:]
        scipy_kdtree.query(row,k)
    construction_time_sum = time.time() - begin_t
    print("利用scipy的kdtree查询10000个点的邻域所需时间：",construction_time_sum)

    print("octree --------------")    
    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum = time.time() - begin_t
    print("建立Octree所需时间：",construction_time_sum)

    result_set=KNNResultSet(k)
    begin_t = time.time()
    for i in range(10000):
        row=db_np[i,:]
        octree.octree_knn_search(root, db_np, result_set, row)
    construction_time_sum = time.time() - begin_t
    print("在Octree利用KNN查询10000个点的邻域所需时间：",construction_time_sum)

    result_set=RadiusNNResultSet(radius)
    begin_t = time.time()
    for i in range(1000):
        row=db_np[i,:]
        octree.octree_radius_search(root, db_np, result_set, row)
    construction_time_sum = time.time() - begin_t
    print("在Octree利用RNN查询1000个点的邻域所需时间：",construction_time_sum)


    
    print("kdtree --------------")    
    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum = time.time() - begin_t
    print("建立kdtree所需时间：",construction_time_sum)
    result_set=KNNResultSet(k)
    begin_t = time.time()
    for i in range(10000):
        row=db_np[i,:]
        kdtree.kdtree_knn_search(root, db_np, result_set, row)
    construction_time_sum = time.time() - begin_t
    print("在kdtree利用KNN查询10000个点的邻域所需时间：",construction_time_sum)
    result_set=RadiusNNResultSet(radius)
    begin_t = time.time()
    for i in range(1000):
        row=db_np[i,:]
        kdtree.kdtree_radius_search(root, db_np, result_set, row)
    construction_time_sum = time.time() - begin_t
    print("在kdtree利用RNN查询1000个点的邻域所需时间：",construction_time_sum)
    """

if __name__ == '__main__':
    main()
