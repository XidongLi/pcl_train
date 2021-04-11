import open3d as o3d
import numpy as np

def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    data_average=data.sum(axis=0)/data.shape[0]
    data_new=data-data_average
    H=data_new.transpose() @ data_new
    eigenvalues,eigenvectors=np.linalg.eig(H)

    # 屏蔽结束
    if sort:
        sort = eigenvalues.argsort()[::-1]
       
        eigenvalues = eigenvalues[sort]

        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

    
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
  w, v = PCA(pc)
  #可视化PCA
  points = [[0,0,0],v[:,0],v[:,1],v[:,2]]
  lines = [[0,1],[0,2],[0,3]]
  line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),lines=o3d.utility.Vector2iVector(lines))
  print('可视化PCA')
  o3d.visualization.draw_geometries([pcd,line_set])
  data_tree = o3d.geometry.KDTreeFlann(pcd)
  #开始计算法向量
  M=list(pc.shape)[0]
  for i in range (0,M):#循环计算每个点
      [k, idx, _]=data_tree.search_knn_vector_3d(pcd.points[i], 10)#确定邻域点
      X=np.asarray(pcd.points)[idx[1:], :]
      w,v=PCA(X) 
      if i==0:
          N=v[:,2] #法向量为PCA中最小特征值对应的特征向量
      else:
          N=np.vstack((N,v[:,2])) #法向量为PCA中最小特征值对应的特征向量，并将其叠加
  #可视化法向量
  point=np.vstack((pc,pc+N)) #pc为当前点，pc+N为法向量的终点
  lines=np.array([range(0,M),range(M,list(point.shape)[0])]).transpose() #确定点的序列组合
  line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(lines))
  print('可视化法向量')
  o3d.visualization.draw_geometries([pcd,line_set])

if __name__ == '__main__':
  main()