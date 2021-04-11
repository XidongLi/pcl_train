import open3d as o3d
import numpy as np
from scipy import spatial

#dataset=np.array([[0,0,0],[0.5,0,0],[1,0,0],[0,0.5,0],[0.5,0.5,0],[1,0.5,0],[0,1,0],[0.5,1,0],[1,1,0],
 #                   [0,0,0.5],[0.5,0,0.5],[1,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[1,0.5,0.5],[0,1,0.5],[0.5,1,0.5],[1,1,0.5],
  #                  [0,0,1],[0.5,0,1],[1,0,1],[0,0.5,1],[0.5,0.5,1],[1,0.5,1],[0,1,1],[0.5,1,1],[1,1,1]])


#print('可视化点云')
#o3d.visualization.draw_geometries([pcd])



def ISS(pc):
    search=spatial.KDTree(pc, leafsize=1)
    L1=[]
    L2=[]
    W=[]
    #求W
    for index_,i in enumerate(pc):
        index=search.query_ball_point(i, 0.5)
        W.append(1/len(index))
    #print('W:',W)
    for index_,i in enumerate(pc):
        index=search.query_ball_point(i, 0.5)
        comatrix=np.zeros((3,3))
        w=0
        #print('当前点：',i)

        for number in index :
            point=pc[number,:]
            #print('对应点：',point)
            #covj=((point-i).reshape(1,3).T@(point-i).reshape(1,3))
            covj=W[number]*((point-i).reshape(1,3).T@(point-i).reshape(1,3))
            #print('wj:',W[number])
            #print('协方差矩阵',covj)
            comatrix=comatrix+covj
            w=w+W[number]
        
        cov_final=comatrix/w
        eigenv,engenvector=np.linalg.eig(cov_final)
        #print('矩阵',cov_final)
        #print('特征值',eigenv)
        #print('特征向量',engenvector)
        
        eigenv=np.sort(eigenv)
        print('特征值',eigenv)
        #print(eigenv)
        #记下有效点
        if eigenv[0]>0.03 :
            L1.append(index_)
            L2.append(eigenv[0])
    #NMS
    L3=[]
    while L2!=[]:
        a=max(L2)
        id_=L2.index(a)
        p_id=L1[id_]
        del L2[id_]
        del L1[id_] 
        L=[]
        for id1,i in enumerate(L1):
            if np.linalg.norm(pc[p_id,:]-pc[i,:]) <=0.5 :
                L.append(id1)
        L.reverse()
        for i in L:
            del L2[i]
            del L1[i]   
        L3.append(p_id)
    return L3
    

def main():
    dataset=np.array([[0,0,0],[0.5,0,0],[1,0,0],[0,0.5,0],[0.5,0.5,0],[1,0.5,0],[0,1,0],[0.5,1,0],[1,1,0],
                    [0,0,0.5],[0.5,0,0.5],[1,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[1,0.5,0.5],[0,1,0.5],[0.5,1,0.5],[1,1,0.5],
                    [0,0,1],[0.5,0,1],[1,0,1],[0,0.5,1],[0.5,0.5,1],[1,0.5,1],[0,1,1],[0.5,1,1],[1,1,1]])
    L1=ISS(dataset)
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(dataset)
    pcd.paint_uniform_color([0, 0, 1])
    
     
    point=dataset[L1[0],:]
    for i in range(1,len(L1)):
        point=np.vstack((point,dataset[L1[i]]))

    print(point)
    pcd_other = o3d.geometry.PointCloud()
    pcd_other.points = o3d.utility.Vector3dVector(point)
    pcd_other.paint_uniform_color([1, 0, 0])
    print('可视化点云')
    o3d.visualization.draw_geometries([pcd_other,pcd])
        
    
  
  
"""
  pcd = o3d.geometry.PointCloud() 
  pcd.points = o3d.utility.Vector3dVector(pc)
  print('可视化原始点云')
  o3d.visualization.draw_geometries([pcd])
  """
  
  
  

if __name__ == '__main__':
  main()