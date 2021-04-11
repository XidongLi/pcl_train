from mydataset import My_dataset
from mymodel import PointNetCls
import torch
import torch.optim as optim
import torch.nn as nn

PATH = '/Users/Mers/Desktop/WS2021/三维点云处理/lesson05/cifar_net.pth'
path = '/Users/Mers/Desktop/modelnet40_normal_resampled'  #path 为 root
data=My_dataset(path)
net=PointNetCls(k=data.__len__(),feature_transform=True)
net.load_state_dict(torch.load(PATH))
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('before device')
net.to(device)
net.train()
print('after device')
dataloader = torch.utils.data.DataLoader(data, batch_size=6)
running_loss = 0.0
for i in range(2000):
    for a, data in enumerate(dataloader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels[:, 0]
        inputs = inputs.transpose(2, 1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward + backward + optimize
        pred, trans, trans_feat = net(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
    print(40*(i+1))
        # print statistics
        
    print(loss.item())

PATH = '/Users/Mers/Desktop/WS2021/三维点云处理/lesson05/cifar_net.pth'
torch.save(net.state_dict(), PATH)




