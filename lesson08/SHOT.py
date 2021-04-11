import numpy as np
from scipy import spatial
import open3d as o3d
import matplotlib.pyplot as plt

def numpy_read_txt(pc_txt_path):
    data = np.genfromtxt(pc_txt_path,delimiter=",")
    #pc = data[:,:6]
    return data

def LRF(fp,pcl,R=0.075):
    pcl_n=pcl[:,3:6]
    pcl_p=pcl[:,:3]
    search=spatial.KDTree(pcl_p, leafsize=1)
    L=[]
    for point in fp:
        index=search.query_ball_point(point[0:3], R)
        W=0
        M=np.zeros((3,3))
        for i in index:
            if np.linalg.norm(point[0:3]-pcl_p[i]) !=0:
                w=R-np.linalg.norm(point[0:3]-pcl_p[i])
                covj=w*((point[0:3]-pcl_p[i]).reshape(1,3).T@(point[0:3]-pcl_p[i]).reshape(1,3))
                M=M+covj
                W=W+w
        M=M/W
        eigenvalues,eigenvectors=np.linalg.eig(M) 
        #print(eigenvectors)
        sort = eigenvalues.argsort()[::-1] 
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        #print(eigenvectors.T)
        L.append(eigenvectors.T)
    #determine x+,z+,y
    P=[]
    for c,point in enumerate(fp):
        index=search.query_ball_point(point[0:3], R)
        #index=search.query_ball_point(fp[0:3], R)
        X_plus=0
        Z_plus=0
        X_minus=0
        Z_minus=0
        for i in index:
            if np.linalg.norm(point[0:3]-pcl_p[i]) !=0:
                if np.dot(L[c][0],point[0:3]-pcl_p[i])>=0:
                    X_plus=X_plus+1
                else :
                    X_minus=X_minus+1
                if np.dot(L[c][2],point[0:3]-pcl_p[i])>=0:
                    Z_plus=Z_plus+1
                else :
                    Z_minus=Z_minus+1
        if X_plus>=X_minus :
            P.append(L[c][0])
        else :
            P.append(-L[c][0])
        if Z_plus>=Z_minus :
            P.append(L[c][0])
        else :
            P.append(-L[c][0])
    P.append(np.cross(P[0],P[1]))
    P.append(np.cross(P[2],P[3]))
    P=[P[0],P[2],P[4],P[5],P[1],P[3]]
    #SHOT
    A=[]
    for a,point in enumerate(fp):
        index=search.query_ball_point(point[0:3], R)
        L=np.zeros((16,5))
        for i in index:
            #print(len(index))
            if np.linalg.norm(point[0:3]-pcl_p[i]) !=0:
                r=(np.linalg.norm(point[0:3]-pcl_p[i])>=R/2)
                x=(np.dot(P[a],point[0:3]-pcl_p[i])>=0)
                y=(np.dot(P[a+2],point[0:3]-pcl_p[i])>=0)
                z=(np.dot(P[a+4],point[0:3]-pcl_p[i])>=0)
                number=8*r+4*x+2*y+z
                #print(number)
                print(pcl_n[i])
                histogram = np.histogram(np.dot(point[3:6],pcl_n[i]), bins=5, range=(-1.0, +1.0))[0]
                #print(histogram)
                L[number]=L[number]+histogram
        L=L.flatten()
        print('aaaaa')
        L=L/L.sum()
        A.append(L)
        
    return A
    
def visual_feature_description(fpfh,keypoint_idx):
    for i in range(len(fpfh)):
        x = [i for i in range(len(fpfh[i]))]
        y = fpfh[i]
        plt.plot(x,y,label=keypoint_idx[i])
    plt.title('Description Visualization for Keypoints')
    plt.legend(bbox_to_anchor=(1, 1),  
               loc="upper left",  
               ncol=1,  
               mode="None",  
               borderaxespad=0,  
               title="keypoints",  
               shadow=False,  
               fancybox=True)  
    plt.xlabel("label")
    plt.ylabel("shot")
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
    L=LRF(fp,np.hstack((np.asarray(downpcd.points),np.asarray(downpcd.normals))),R=0.075)
    visual_feature_description(L,fp[:,:3])
    
if __name__ == '__main__':
    main()
                
        