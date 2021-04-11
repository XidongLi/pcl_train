import numpy as np
import math
import open3d as o3d
from scipy import spatial

def numpy_read_txt(pc_txt_path):
    data = np.genfromtxt(pc_txt_path,delimiter=",")
    #pc = data[:,:6]
    return data

def main():
    path = "C:/Users/Mers/Desktop/pc/table_0001.txt"
    pc = numpy_read_txt(path)#6维数据
    pcd = o3d.geometry.PointCloud() 
    fp=np.array([[0.62,-0.3,0.62],[-0.62,-0.3,0.62]])
    fp_= o3d.geometry.PointCloud() 
    fp_.points = o3d.utility.Vector3dVector(fp)
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([downpcd,fp_])
    downpcd.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(downpcd.points),fp)))
    downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print(np.asarray(downpcd.points)[-1])
    print(np.asarray(downpcd.normals)[-1])
    print(np.asarray(downpcd.points)[-2])
    print(np.asarray(downpcd.normals)[-1])
    #print('aaaaaaaa',np.asarray(downpcd.normals))
    
    
          
   
    
    
    
    


    
    
    
    
    
if __name__ == '__main__':
    main()