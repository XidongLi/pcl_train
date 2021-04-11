import open3d as o3d
import numpy as np
from scipy import spatial



def numpy_read_txt(pc_txt_path):
  data = np.genfromtxt(pc_txt_path,delimiter=",")
  pc = data[:,:3]
  return pc


def ISS(pc):
    search=spatial.KDTree(pc, leafsize=10)
    L1=[]
    L2=[]
    W=[]
    #求W
    for index_,i in enumerate(pc):
        index=search.query_ball_point(i, 0.5)
        W.append(1/len(index))
    for index_,i in enumerate(pc):
        index=search.query_ball_point(i, 0.5)
        if len(index)<=5:
            print('r too small')
        comatrix=np.zeros((3,3))
        w=0
        #w=len(index)
        for number in index :
            point=pc[number,:]
            #covj=((point-i).reshape(1,3).T@(point-i).reshape(1,3))
            covj=W[number]*((point-i).reshape(1,3).T@(point-i).reshape(1,3))
            comatrix=comatrix+covj
            w=w+W[number]
        cov_final=comatrix/w
        eigenv,engenvector=np.linalg.eig(cov_final)
        #print(eigenv)
        eigenv=np.sort(eigenv)
        #print(eigenv)
        #记下有效点
        if eigenv[0]>0.00005 and (eigenv[1]/eigenv[2])<0.8 and (eigenv[0]/eigenv[1])<0.8 :
            L1.append(index_)
            L2.append(eigenv[0])
    #return L1

    #NMS
    L3=[]
   
    while L2!=[]:
        
        a=max(L2)
        #print('a',a)
        id_=L2.index(a)
        #print('id_',id_)
        p_id=L1[id_]
        #print('p_id',p_id)
        del L2[id_]
        del L1[id_]
        
        L=[]
        for id1,i in enumerate(L1):
            
            if np.linalg.norm(pc[p_id,:]-pc[i,:]) <0.2 :
                L.append(id1)
        
        L.reverse()
     
        for i in L:
            del L2[i]
            del L1[i]
     
        L3.append(p_id)
    return L3
    
    
            
            
            
             
def main():
    path = "C:/Users/Mers/Desktop/pc/guitar_0060.txt"
    pc = numpy_read_txt(path)
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])
    downpcd.paint_uniform_color([0, 0, 1])
    
    
    L1=ISS(np.asarray(downpcd.points))
    print(len(L1))
    
    point=pc[L1[0],:]
    for i in range(1,len(L1)):
        point=np.vstack((point,pc[L1[i]]))
    #pc=np.delete(np.asarray(downpcd.points), L1, 0)
    
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(point)
    pcd.paint_uniform_color([1, 0, 0])
    print('可视化点云')
    o3d.visualization.draw_geometries([downpcd,pcd])
        
    
  
  
"""
  pcd = o3d.geometry.PointCloud() 
  pcd.points = o3d.utility.Vector3dVector(pc)
  print('可视化原始点云')
  o3d.visualization.draw_geometries([pcd])
  """
  
  
  

if __name__ == '__main__':
  main()