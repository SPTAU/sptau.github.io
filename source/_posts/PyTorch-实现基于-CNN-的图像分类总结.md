---
title: PyTorch 实现基于 CNN 的图像分类总结
mathjax: true
categories:
  - - 技术
    - Python
tags:
  - Python
  - PyTorch
description: 本次实验采用 MNIST 作为数据集，搭建 CNN 实现手写数字识别
abbrlink: 605ef827
date: 2022-09-12 19:52:36
---

## 数据集

```py
from torchvision import datasets, transforms

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root="./dataset/mnist/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root="./dataset/mnist/", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, atch_size=batch_size)
```

### torchvision.datasets

torchvision.datasets 中包含了 `MNIST` 数据集

通过 torchvision.datasets.MNIST 提供的 API 可以下载、转换、加载数据集

```py
datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

#### 参数解析

1. root

    str ，数据集路径对象

    用于储存 `processed/training.pt` 和 `processed/test.pt`

2. train

    bool ，可选

    - 当 train 参数为 True 时，载入的数据集为训练集
    - 当 train 参数为 False 时，载入的数据集为测试集

3. download

    bool ，可选

    - 当 download 参数为 True 且未下载过该数据时，将从 <http://yann.lecun.com/exdb/mnist/> 下载数据集并将其放在 root 参数指定路径下

    - 如果 download 参数为 True 且下载过该数据，则不会再次下载

4. transform

    可调用对象，可选

    接收 PIL 映像并返回转换版本的函数/变换。例如:transforms.RandomCrop

5. target_transform

    可调用对象，可选

    一个接收目标并转换它的函数/变换。

    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version. E.g, transforms.RandomCrop
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.

### torchvision.transforms

torchvision.transforms 可以实现对图像的变化

#### torchvision.transforms.Compose(transforms)

Compose  函数可以实现不同变化的组合

#### torchvision.transforms.ToTensor()

ToTensor  函数可以将图像的多通道矩阵转换为张量形式，方便后续计算

#### torchvision.transforms.Nomalize(mean, std[, inplace])

Normalize 函数可以用均值和标准差对张量图像进行归一化

##### MNIST 数据集的均值和标准差

MNIST 数据集的均值为 0.1307 ， 标准差为 0.3081

## 网络设计

```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### `torch.nn.Conv2d()` 函数

Conv2d 函数可以实现卷积层

```py
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

#### 部分参数解析

1. in_channels

    int ，必选

    输入的通道数

2. out_channels

    int ，必选

    输出的通道数

3. kernel_size

    int 或 tuple ，必选

    卷积核的尺寸

4. stride

    int 或 tuple ，可选，默认为 1

    卷积时的步长

5. padding

    int 或 tuple ，可选，默认为 0

    填充边缘的宽度

6. dilation

    int 或 tuple ，可选，默认为 1

    卷积核中各元素的间距

7. padding_mode

    str ，可选，默认为 'zeros'

    选择填充模式

    可选 'zeros' ， 'reflect' ， 'replicate' ， 'circular'

#### 参考网页

Pytorch - [CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

夏普通 - [pytorch之torch.nn.Conv2d()函数详解](https://blog.csdn.net/qq_34243930/article/details/107231539)

### `torch.nn.MaxPool2d()` 函数

MaxPool2d 函数可以实现最大池化层

```py
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

#### 部分参数解析

1. kernel_size

    int 或 tuple ，必选

    进行最大值池化的窗口大小

2. stride

    int 或 tuple ，可选，默认与 kernel_size 参数相同

    最大值池化窗口的步长

3. padding

    int 或 tuple ，可选，默认为 0

    填充边缘的宽度

#### 参考网页

PyTorch - [MAXPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)

### `torch.nn.Linear()` 函数

Linear 函数可以实现全连接层，其本质上是线性层

```py
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

#### 部分参数解析

1. in_features

    输入数据的维度

2. out_features

    输出数据的维度

3. bias

    bool ，默认为 True

    决定是否开启加法偏置

#### 参考网页

PyTorch - [LINEAR](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear)

### PyTorch 使用 GPU 训练

PyTorch 使用 GPU 训练时先需要

- 单 GPU 训练时

    ```py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```

- 多 GPU 训练时

    需要根据所需使用的 GPU ，在文件头部添加

    ```py
    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"
    ```

    使得当前代码仅对指定 GPU 可见，系统将会对指定 GPU 从零开始重新编号

    ```py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

- 训练过程中

    需要将模型和张量计算都搬运到 GPU 上

    ```py
    model.to(device)
    ```

    ```py
    inputs, targets = inputs.to(device), targets.to(device)
    ```

#### 参考网页

hello_dear_you - [pytorch 之多 GPU 训练](https://blog.csdn.net/hello_dear_you/article/details/120190567)

## 设计 Loss 和优化器

```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```
