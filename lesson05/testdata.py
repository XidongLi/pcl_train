from mydataset import My_dataset
from mymodel import PointNetCls
import torch


path = '/Users/Mers/Desktop/modelnet40_normal_resampled'
PATH = '/Users/Mers/Desktop/WS2021/三维点云处理/lesson05/cifar_net.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data=My_dataset(root=path,split='test')
net=PointNetCls(k=data.__len__(),feature_transform=True)
net.load_state_dict(torch.load(PATH))
net.eval()
net.to(device)
print('aaaaa')
testloader = torch.utils.data.DataLoader(data, batch_size=1)


total_correct = 0
total_testset = 0
with torch.no_grad():
    for i in range(10):
        for i,data in enumerate(testloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            pred, trans, trans_feat = net(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        print("final accuracy {}".format(total_correct / float(total_testset)))
print("final accuracy {}".format(total_correct / float(total_testset)))


