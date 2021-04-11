import numpy as np
import math
import open3d as o3d
from scipy import spatial
import matplotlib.pyplot as plt

def numpy_read_txt(pc_txt_path):
    data = np.genfromtxt(pc_txt_path,delimiter=",")
    #pc = data[:,:6]
    return data

def Quadruplet(p1,p2,n1,n2):
    u=n1
    v=np.cross(u,(p2-p1)/np.linalg.norm(p2-p1))
    w=np.cross(u,v)
    alpha=np.dot(v,n2)
    phi=np.dot(u,(p2-p1)/np.linalg.norm(p2-p1))
    xita=math.atan(np.dot(w,n2)/np.dot(u,n2))
    
    return np.array([alpha,phi,xita])

def SPFH_Histogram(SPFH,B):
    alpha_histogram = np.histogram(SPFH[:,0], bins=B, range=(-1.0, +1.0))[0]
    alpha_histogram = alpha_histogram / alpha_histogram.sum()
    
    phi_histogram = np.histogram(SPFH[:,1], bins=B, range=(-1.0, +1.0))[0]
    phi_histogram = phi_histogram / phi_histogram.sum()
    
    theta_histogram = np.histogram(SPFH[:,2], bins=B, range=(-np.pi, +np.pi))[0]
    theta_histogram = theta_histogram / theta_histogram.sum()
   
    signature = np.hstack((alpha_histogram,phi_histogram,phi_histogram))
    return signature

def FPFH(fp,r,pcl,B):#fp是六维
    pcl_n=pcl[:,3:6]
    pcl_p=pcl[:,:3]
    search=spatial.KDTree(pcl_p, leafsize=1)
    L=[]
    for point in fp:
       
        FPFH=np.array([0,0,0])
       
        SPFH_pq=np.array([0,0,0])
        
        index=search.query_ball_point(point[0:3], 0.075)
        SPFH_pk_final=np.zeros((3*B))
        
        for i in index:
            if np.linalg.norm(point[0:3]-pcl_p[i]) !=0:
                SPFH_pq=np.vstack((SPFH_pq,Quadruplet(point[0:3],pcl_p[i],point[3:6],pcl_n[i])))
                SPFH_pk=np.array([0,0,0])
                d,index_2=search.query(pcl_p[i], k=len(index))
                for a in index_2:
                    if np.linalg.norm(pcl_p[i]-pcl_p[a]) !=0:
                        SPFH_pk=np.vstack((SPFH_pk,Quadruplet(pcl_p[i],pcl_p[a],pcl_n[i],pcl_n[a])))
                wk=1/np.linalg.norm(point[0:3]-pcl_p[i])
                SPFH_pk=SPFH_Histogram(SPFH_pk[1:len(index)+1,:],B)*wk
                SPFH_pk_final=SPFH_pk_final+SPFH_pk
        SPFH_pk_final=SPFH_pk_final/(len(index)-1)
        SPFH_pq=SPFH_pq[1:len(index)+1,:]
        SPFH_pq_histogramm=SPFH_Histogram(SPFH_pq,B)
        spfh = SPFH_pq_histogramm + SPFH_pk_final
        spfh = spfh / np.linalg.norm(spfh)
        L.append(spfh)
    return L
    #print(L)
def visual_feature_description(fpfh,keypoint_idx):
    for i in range(len(fpfh)):
        x = [i for i in range(len(fpfh[i]))]
        y = fpfh[i]
        plt.plot(x,y,label=keypoint_idx[i])
    plt.title('Description Visualization for Keypoints')
    plt.legend(bbox_to_anchor=(1, 1),  
               loc="upper right",  
               ncol=1,  
               mode="None",  
               borderaxespad=0,  
               title="keypoints",  
               shadow=False,  
               fancybox=True)  
    plt.xlabel("label")
    plt.ylabel("fpfh")
    plt.show()
                


def main():
    path = "C:/Users/Mers/Desktop/pc/table_0001.txt"
    pc = numpy_read_txt(path)#6维数据
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    fp=np.array([[0.62,-0.3,0.62],[-0.62,-0.3,0.62]])
    downpcd.points=o3d.utility.Vector3dVector(np.vstack((np.asarray(downpcd.points),fp)))
    fp=np.array([[0.62,-0.3,0.62,-0.22668157,-0.97377525,0.01942234],[-0.62,-0.3,0.62,-0.22668157,-0.97377525,0.01942234]])
    downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #print('aaaaaaaa',np.asarray(downpcd.normals))
    L=FPFH(fp,0.15,np.hstack((np.asarray(downpcd.points),np.asarray(downpcd.normals))),5)
    visual_feature_description(L,fp[:,:3])
    

    
    
if __name__ == '__main__':
    main()


