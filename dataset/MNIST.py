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
    image = image.contiguous().view(-1,c)
    image = image[permutation,:]
    image = image.view(c,h,w)
    return image
class MNISTTask():
    def __init__(self,root,task_num=1):
        self.root = root
        self.task_num = task_num
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.3081), (0.3081))
            # Cutout(n_holes=1, length=16)
        ])
        self.transform = transforms.Compose(
            [transforms.Normalize((0.3081), (0.3081))]
        )
        np.random.seed(20)
        self.permutations = [np.random.permutation(784) for i in range(task_num-1)]
    def getTaskDataSet(self):
        trainDataset = myMNIST(root=self.root,train=True ,transform=self.transform_train, download=False, permutations=self.permutations)
        train_task_datasets = [MNISTDataset(data, self.transform_train) for data in trainDataset.task_datasets]
        testDataset = myMNIST(root=self.root, train=False, transform=self.transform, download=False, permutations=self.permutations)
        test_task_datasets = [MNISTDataset(data, self.transform) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets

class MNISTDataset(Dataset):
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

class myMNIST(datasets.MNIST):
    def __init__(self,
            root: str,
            train=True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            permutations=None):
        super(myMNIST, self).__init__(root,train=train,transform=transform,target_transform=target_transform,download=download)
        basedataset=[]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        for img,label in zip(self.data,self.targets):
            d= {}
            d[0]=transform_train(Image.fromarray(img.numpy(), mode='L'))
            d[1]=label
            basedataset.append(d)


        self.task_datasets = [[{0:da[0].clone().detach(),1:da[1]} for da in basedataset]]

        for i,permutation in enumerate(permutations):
            datas = []
            for b in basedataset:
                d = {}
                d[0] = permutation_image(b[0].clone().clone().detach(),permutation)
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
        labels = labels-10
        pred = model(images.to(device),0)
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    acc = sum_num.item() / num_samples

    return acc

data = MNISTTask('../../data',task_num=5)
train_dataset,test_dataset = data.getTaskDataSet()
# train_dataset = datasets.SVHN('../data/SVHN',split='train',download=False,transform=transforms.ToTensor())
# test_dataset = datasets.SVHN('../data/SVHN',split='test',transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=128,
#                                            shuffle=True,
#                                            pin_memory=True,
#                                            num_workers=0)
#
# val_loader = torch.utils.data.DataLoader(test_dataset,
#                                          batch_size=64,
#                                          shuffle=False,
#                                          pin_memory=True,
#                                          num_workers=0)

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

# transforms = transforms.Compose(
#
#             [   transforms.ToTensor(),
#                 transforms.Normalize(0.1307, 0.3081)]
#         )
net_glob = RepTail([1,28,28]).cuda()
# train_dataset = datasets.MNIST(root='../data',train=True,transform=transforms,download=False)
# test_dataset = datasets.MNIST(root='../data',train=False,transform=transforms,download=False)
train_loader = torch.utils.data.DataLoader(train_dataset[1],
                                           batch_size=128,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0)

val_loader = torch.utils.data.DataLoader(test_dataset[1],
                                         batch_size=64,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0)

opt = torch.optim.Adam(net_glob.parameters(), 0.001)
ce = torch.nn.CrossEntropyLoss()
# print(net_glob.weight_keys)
for epoch in range(100):
    net_glob.train()
    for x, y in train_loader:
        x = x.cuda()
        y = y.cuda() - 10
        out = net_glob(x,0,is_cifar=False)
        loss = ce(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 1 == 0:
        acc = evaluate(net_glob, val_loader, 'cuda:0')
        print('The epochs:' + str(epoch) + '  the acc:' + str(acc))


