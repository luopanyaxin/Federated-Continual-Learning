# 联邦持续学习
## 1 介绍
联邦学习作为一个机器学习框架，能够使不同地点的设备在满足用户隐私等要求下，一起进行数据建模和模型构建。最近，人们希望机器能够像人类一样不断学习新的知识而不会对旧的知识产生灾难性遗忘。而在现实生活中对一个新知识的学习往往是通过和其他人一起互相交流学习而来，例如：书籍，视频等。因此，本课题将利用联邦学习的框架和持续学习相结合，使得各个设备能够互相学习，借鉴彼此已有的知识增强训练效果。
## 2 代码相关
### 2.1 安装
**相关准备**
- Windows 
- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ 

**准备虚拟环境**
1. 准备conda环境并进行激活.
	```shell
	conda create -n FedCL python=3.7
	conda active FedCL
	```
2. 在[官网](https://pytorch.org/)安装对应版本的pytorch
![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0452f4e390ca4e7b9e9dce60f9a40c58~tplv-k3u1fbpfcp-zoom-1.image)
可以在终端直接使用官方提供的code
   
3.  安装FedCL所需要的包
	```shell
	git clone https://github.com/luopanyaxin/Federated-Continual-Learning
	pip install -r requirements.txt
	```
 **Note:  安装前请确保电脑上是否有显卡且显卡算力和pytorch版本匹配**
### 2.2 运行
FedKNOW 根据下面的命令进行运行：
```shell
python main_WEIT.py --dataset [dataset] --model [mdoel] --num_users [num_users]  
--shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] 
--task [task] --epoch [epoch]  --local_ep [local_ep]  --gpu [gpu]
```
参数解释：
- `dataset` : 数据集，例如：`cifar100`,`FC100`,`CORe50`,`SVHN`, `mnist`
- `model`: 网络模型，例如：`6-Layers CNN`, `ResNet18`
- `num_users` : 客户端数量
- `shard_per_user`: 每个客户端拥有的类
- `frac`：每一轮参与训练的客户端
- `local_bs`：每一个客户端的batch_size
- `lr`：学习率
- `task`：任务数
- `epochs`: 客户端和服务器通信的总次数
- `local_ep`：本地客户端迭代数
- `gpu`：GPU ID

完整的参数信息解释在`utils/option.py`。
## 3 实验细节描述
### 3.1 实验设置
**数据集:**
- Cifar100: Cifar100共包含50000条训练数据和10000条测试数据，共100个类。在持续学习中，我将其分为10个任务，每个任务包含10类。
- FC100: FC100共包含50000条训练数据和10000条测试数据，共100个类。在持续学习中，我将其分为10个任务，每个任务包含10类。
- MNIST: MNIST数据集包含60000条训练数据和10000条测试数据，共10个类。在持续学习中，将这些数据按照随机序列进行重新排序生成5个任务，每个任务包括10类。
- SVHN: SVHN数据集包含73257条训练数据和26032条测试数据，共10个类，为了使各个类的数据数量一致，在训练集中每个类选取4500个数据样本，在测试集中选取1500个测试样本。在持续学习中，将这些数据按照随机序列进行重新排序生成5个任务，每个任务包括10类。
在联邦学习中，每个任务数据我们利用non-iid的方式分配给20个客户端。
**模型:**
- 6_layer CNN: 参考[AGS-CL](https://arxiv.org/abs/2003.13726)实现的模型，共包含6个卷积层和2个全连接层。
- ResNet : 参考pytorch官方提供的resnet18进行的改动。
**Baseline:**
- GEM: 持续学习中的经典算法，通过存储部分样本后通过梯度的旋转来防止遗忘。
- Co2L: 2021年提出的最新夫人持续学习算法，通过使用对比学习目标学习表征，再通过自监督蒸馏方式来保留表征方式来防止遗忘。
- FedAvg: 联邦学习经典算法，将各个客户端的参数加权平均算法。
- APFL: 个性化联邦学习算法，通过设定参数比例来权衡各个客户端模型参数聚合参数，防止数据异构导致的算法发散。
- FedRep: 个性化联邦学习，划分每个模型参与全局聚合的层以及本地训练的层，之后通过冻结梯度的方式依次更新对应部分的层参数，防止数据异构导致的算法发散。
**实验设置表：**
    |Devices|Models and data|Baselines|
    |--|--|--|
    |<br>Windows <br>|6-layer CNN on CIFAR100<br>6-layer CNN on FC100<br>6-layer CNN on MNIST<br>ResNet18 on SVHN|GEM<br>Co2L<br>AGS-CL<br>FedAvg<br>APFL<br>FedRep

### 3.2 Experiment code
- 6-layer CNN on Cifar100
	```shell
	python main_WEIT.py --epochs=150 --round=15 --num_users=20 --frac=0.4 shard_per_user=5 per--model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 lr=0.001
	```
- 6-layer CNN on FC100
	```shell
	python main_WEIT.py --epochs=150 --round=15 --num_users=20 --frac=0.4 shard_per_user=5 per--model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 lr=0.001
	```
- 6-layer CNN on MNIST
	```shell
	python main_WEIT.py --epochs=75 --round=15 --num_users=20 --frac=0.4 shard_per_user=5 per--model=6_layerCNN --dataset=mnist --num_classes=100 --task=5 lr=0.001
	```
- ResNet18 on SVHN
	```shell
	python main_WEIT.py --epochs=75 --round=15 --num_users=20 --frac=0.4 shard_per_user=5 per--model=6_layerCNN --dataset=SVHN --num_classes=100 --task=5 lr=0.001
	```

#### 3.3 Experiment result 
##### 3.3.1 Cifar100
- 准确率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/cifar100.png)
- 遗忘率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/cifar100_fr.png)
##### 3.3.1 FC100
- 准确率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/FC100.png)
- 遗忘率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/FC100_fr.png)
##### 3.3.1 MNIST
- 准确率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/MNIST.png)
- 遗忘率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/MNIST_fr.png)
##### 3.3.1 SVHN
- 准确率图像
     ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/SVHN.png)
- 遗忘率图像
     ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/SVHN_fr.png)



