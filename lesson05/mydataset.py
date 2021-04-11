import os
import torch.utils.data as data
import random
import numpy as np
import torch


def numpy_read_txt(pc_txt_path):
  data = np.genfromtxt(pc_txt_path,delimiter=",")
  pc = data[:,:3]
  return pc
            
class My_dataset(data.Dataset):
    def __init__(self,root,split='train'):
        self.root = root
        self.split = split
        self.fns = []
        with open(os.path.join(root, 'modelnet40_shape_names.txt'), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
        
        self.cat = {}
        for class_,object_ in enumerate(self.fns):
            self.cat[object_] = int(class_)
       
        self.data_num=[]
        with open(os.path.join(root,'modelnet40_{}.txt'.format(self.split))) as b:
            for line in b:
                self.data_num.append(line.strip())
     
        self.index2=[]
        c=0
        for object_ in self.fns:
            L=[]
            
            if len(self.data_num[c].split('_'))==2 :
                while self.data_num[c].split('_')[0]==object_  and c<=len(self.data_num):
                    L.append(c)
                    c=c+1
                    if c==len(self.data_num):
                        self.index2.append(L)
                        break
            
            elif len(self.data_num[c].split('_'))>2:
                while  self.data_num[c].split('_')[0]==object_.split('_')[0] and c<=len(self.data_num):
                    L.append(c)
                    c=c+1
                
            self.index2.append(L)


    def __getitem__(self, index):

        fn = self.fns[index]
        class_ = self.cat[fn]
        #with open(os.path.join(root, fn,self.data_num[int(random.sample(self.index2[index], 1))]), 'r') as f:

        #print(fn)
        
        #print(random.sample(self.index2[index], 1))

        point_set= numpy_read_txt(os.path.join(self.root, fn,self.data_num[random.sample(self.index2[index], 1)[0]]+'.txt'))
        point_set = torch.from_numpy(point_set.astype(np.float32))
        class_ = torch.from_numpy(np.array([class_]).astype(np.int64))
       
        return point_set, class_
        
    def __len__(self):
        return len(self.fns)
            


    

