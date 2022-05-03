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
### 3.1 Experiment setting
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
##### 3.3.1 FC100
- 准确率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/FC100.png)
- 遗忘率图像
##### 3.3.1 MNIST
- 准确率图像
    ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/MNIST.png)
- 遗忘率图像
##### 3.3.1 SVHN
- 准确率图像
     ![image](https://github.com/luopanyaxin/Federated-Continual-Learning/blob/main/Experiment/SVHN.png)
- 遗忘率图像


