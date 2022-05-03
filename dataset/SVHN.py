import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from models.Nets import RepTail
from torch.utils.data import DataLoader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

def permutation_image(image,permutation=None):
    if permutation is None:
        return image
    c,h,w = image.size()
    image = image.contiguous().view(c,-1)
    for i in range(c):
        image[i] = image[i][permutation]
    image = image.view(c,h,w)
    return image
class SVHNTask():
    def __init__(self,root,task_num=1):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.root = root
        self.task_num = task_num
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
            # Cutout(n_holes=1, length=16)
        ])
        self.transform = transforms.Compose(
            [transforms.Normalize(mean, std)]
        )
        np.random.seed(20)
        self.permutations = [np.random.permutation(1024) for i in range(task_num-1)]
    def getTaskDataSet(self):
        trainDataset = mySVHN(root=self.root, split='train', transform=self.transform_train, download=True, permutations=self.permutations,maxdata=4500)
        train_task_datasets = [SVHNDataset(data, self.transform_train) for data in trainDataset.task_datasets]
        testDataset = mySVHN(root=self.root, split='test', transform=self.transform, download=True, permutations=self.permutations,maxdata=1500)
        test_task_datasets = [SVHNDataset(data, self.transform) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets

class SVHNDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][0], self.data[index][1]

        # doing this so that it is consistent with all other dataset

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class mySVHN(datasets.SVHN):
    def __init__(self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            maxdata = 0,
            permutations=None):
        super(mySVHN, self).__init__(root,split,transform,target_transform,download)
        classes = {}
        basedataset=[]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        for i in range(10):
            classes[i] = 0
        for img,label in zip(self.data,self.labels):
            if classes[label] < maxdata:
                classes[label] += 1
                d= {}
                d[0]=transform_train(Image.fromarray(np.transpose(img)))
                transforms.ToTensor()
                d[1]=label
                basedataset.append(d)


        self.task_datasets = [[{0:da[0],1:da[1]} for da in basedataset]]

        for i,permutation in enumerate(permutations):
            datas = []
            for b in basedataset:
                d = {}
                d[0] = permutation_image(b[0],permutation)
                d[1] = b[1] + (i+1)*10
                datas.append(d)
            self.task_datasets.append(datas)
    def __len__(self):
        return len(self.task_datasets[0])

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device),0)
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    acc = sum_num.item() / num_samples

    return acc

# data = SVHNTask('../data/SVHN',task_num=5)
# train_dataset,test_dataset = data.getTaskDataSet()
train_dataset = datasets.SVHN('../data/SVHN',split='train',download=False,transform=transforms.ToTensor())
test_dataset = datasets.SVHN('../data/SVHN',split='test',transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=128,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0)

val_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=64,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0)

# net_glob = RepTail([3,32,32])
# net_glob.cuda()
# opt = torch.optim.Adam(net_glob.parameters(), 0.001)
# ce = torch.nn.CrossEntropyLoss()
# # print(net_glob.weight_keys)
# for epoch in range(100):
#     net_glob.train()
#     for x, y in train_loader:
#         x = x.cuda()
#         y = y.cuda()
#         out = net_glob(x,0,is_cifar=False)
#         loss = ce(out, y)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#     if epoch % 1 == 0:
#         acc = evaluate(net_glob, val_loader, 'cuda:0')
#         print('The epochs:' + str(epoch) + '  the acc:' + str(acc))

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# define hyper parameters
Batch_size = 50
Lr = 0.01
Epoch = 10

train_dataset = torchvision.datasets.SVHN(
    root='../data/SVHN',
    split='train',
    download=False,
    transform=torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.SVHN(
    root='../data/SVHN',
    split='test',
    download=False,
    transform=torchvision.transforms.ToTensor()
)

# define train loader
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=Batch_size
)

# define test loader
test_loader = Data.DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=Batch_size
)


# 取一个固定测试点作为测试数据
# dataiter = iter(test_loader)
# test_x, test_y = dataiter.next()

# construct network

ResNet18 = RepTail()
# torchvision.models.resnet18(pretrained=False).cuda()
# print(ResNet18)

opt = torch.optim.SGD(ResNet18.parameters(), lr=Lr)
loss_fun = nn.CrossEntropyLoss()
a = []
ac_list = []
for epoch in range(Epoch):
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        output = ResNet18(x,t=0)

        loss = loss_fun(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = evaluate(ResNet18, test_loader, 'cuda:0')
    print(acc)


# print('real value', test_y[: 10].numpy())
# print('train value', torch.max(ResNet18(test_x)[: 10], dim=1)[1].numpy())

plt.plot(a, ac_list, color='r')
plt.show()

# d = {}
# for i in range(10):
#     d[i]=0
# # for l in data.labels:
#     d[l]+=1
# for x,y in data:
#     a=1
# def main():
#
#
# if __name__ == "__main__":
#     main()
