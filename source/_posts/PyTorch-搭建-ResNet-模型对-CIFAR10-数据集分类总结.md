---
title: PyTorch 搭建 ResNet 模型对 CIFAR10 数据集分类总结
mathjax: true
categories:
  - - 技术
    - 学习
tags:
  - Python
  - PyTorch
  - CNN
description: >-
  本次试验将以 [Deep Residual Learning for Image Recognition] 作为主要参考来源，使用 PyTorch 搭建
  ResNet 模型实现对 CIFAR10 数据集的训练和测试
abbrlink: cfb33f60
date: 2022-09-18 21:28:12
---

## ResNet 模型搭建

### 结构分析

#### 通用框架

根据 [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) 论文中的信息，可以得到常规 ResNet 模型的通用框架如下

| layer_name | out_size | (18/34 layers) out_channel | (50/101/152 layers) out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Input | 224\*224 | 3 | 3 | None | None | None |
| Conv1 | 112\*112 | 64 | 64 | 7 | 2 | 3 |
| Maxpool | 56\*56 | 64 | 64 | 3 | 2 | 1 |
| Conv2_x | 56\*56 | 64 | 64*4=256 | - | - | - |
| Conv3_x | 28\*28 | 128 | 128*4=512 | - | - | - |
| Conv4_x | 14\*14 | 256 | 256*4=1024 | - | - | - |
| Conv5_x | 7\*7 | 512 | 512*4=2048 | - | - | - |
| Avgpool | 1\*1 | 512 | 2048 | None | None | None |
| Flatten | 2048 | 1 | 1 | None | None | None |
| FC | 1000 | 1 | 1 | None | None | None |

其中 Conv2_x 、 Conv3_x  、Conv4_x 、 Conv5_x 层可由 BasicBlock 和 Bottleneck 两种基本模型组合而成

| layer_name | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Conv2_x | BasicBlock\*2 | BasicBlock\*3 | Bottleneck\*3 | Bottleneck\*3 | Bottleneck\*3 |
| Conv3_x | BasicBlock\*2 | BasicBlock\*4 | Bottleneck\*4 | Bottleneck\*4 | Bottleneck\*8 |
| Conv4_x | BasicBlock\*2 | BasicBlock\*6 | Bottleneck\*6 | Bottleneck\*23 | Bottleneck\*36 |
| Conv5_x | BasicBlock\*2 | BasicBlock\*3 | Bottleneck\*3 | Bottleneck\*3 | Bottleneck\*3 |

#### 基础结构

ResNet 18/34 使用的 BasicBlock 结构如下

| layer_name | in_size | out_size |  out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Conv1 | x\*x | (x/stride)\*(x/stride) | out_channel | 3 | stride | 1 |
| Conv2 | x'\*x' | x'\*x' | out_channel | 3 | 1 | 1 |
| identity |

ResNet 50/101/152 使用的 Bottleneck 结构如下

| layer_name | in_size | out_size |  out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Conv1 | x\*x | (x/stride)\*(x/stride) | out_channel | 1 | 1 | 0 |
| Conv2 | x'\*x' | x'\*x' | out_channel | 3 | stride | 1 |
| Conv3 | x'\*x' | x'\*x' | out_channel | 1 | 1 | 0 |
| identity |

#### stride 和 identity

当基础结构是 Conv3_x 、Conv4_x 、 Conv5_x 的第一层时， `stride=2` 且 identity 为下采样后的输入

```py
nn.Sequential(
    nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
    nn.BatchNorm2d(out_channel * block.expansion),
)
```

当基础结构是 Conv3_x 、Conv4_x 、 Conv5_x 的其他层或在 Conv2_x 层时， `stride=1` 且 identity 为输入本身

```py
nn.Sequential()
```

### 网络实现

#### BasicBlock

```py
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # transforming (batch_size * x * x * input_channel) to (batch_size * x * x * output_channel)
        #                                                   or (batch_size * x/2 * x/2 * output_channel)
        # floor(((x - 3 + 2 * 1) / stride) + 1) => floor(x) stride = 1
        #                                       => floor(x/2) stride = 2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x' * x' * output_channel) to (batch_size * x' * x' * output_channel)
        # floor(((x' - 3 + 2 * 1) / 1) + 1) => floor(x')
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
```

#### Bottleneck

```py
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # transforming (batch_size * x * x * output_channel) to (batch_size * x * x * output_channel)
        # floor(((x - 3 + 2 * 1) / 1) + 1) => floor(x)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x * x * output_channel) to (batch_size * x * x * output_channel)
        #                                                    or (batch_size * x/2 * x/2 * output_channel)
        # floor(((x - 3 + 2 * 1) / stride) + 1) => floor(x) stride = 1
        #                                       => floor(x/2) stride = 2
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x' * x' * output_channel) to (batch_size * x' * x' * (output_channel* expansion))
        # floor(((x' - 3 + 2 * 1) / 1) + 1) => floor(x')
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
```

#### 通用框架

```py
class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], num_block: List[int], num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channel = 64
        # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 112 * 112 * 64)
        # floor(((224 - 7 + 2 * 3) / 2) + 1) => floor(112.5) => floor(112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # transforming (batch_size * 112 * 112 * 64) to (batch_size * 56 * 56 * 64)
        # floor(((112 - 3 + 2 * 1) / 2) + 1) => floor(56.5) => floor(56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # transforming (batch_size * 56 * 56 * 64) to (batch_size * 56 * 56 * (64 * block.expansion))
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride=1)
        # transforming (batch_size * 56 * 56 * (64 * block.expansion)) to (batch_size * 28 * 28 * (128 * block.expansion))
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2)
        # transforming (batch_size * 28 * 28 * (128 * block.expansion)) to (batch_size * 14 * 14 * (256 * block.expansion))
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2)
        # transforming (batch_size * 14 * 14 * (256 * block.expansion)) to (batch_size * 7 * 7 * (512 * block.expansion))
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2)
        # transforming (batch_size * 7 * 7 * (512 * block.expansion)) to (batch_size * 1 * 1 * (512 * block.expansion))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # transforming (batch_size * 2048) to (batch_size * num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], out_channel: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

#### 构造网络

```py
def ResNet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3])
```

#### 参考网页

PyTorch - [SOURCE CODE FOR TORCHVISION.MODELS.RESNET](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet101)

明素 - [ResNet详解](https://blog.csdn.net/weixin_43406381/article/details/118404612)

### 回顾

#### `nn.Conv2d()` 函数 `bias` 参数的设置

当 `nn.Conv2d()` 后接 `nn.BatchNorm2d()` 时，可以把 `bias` 参数设置为 `False`

因为在 BN 层中，输入是否存在偏置不影响输出结果

不添加偏置还可以减少显卡内存的占用

##### 参考网页

7s记忆的鱼 - [【pytorch】Conv2d()里面的参数bias什么时候加，什么时候不加？](https://blog.csdn.net/qq_38230414/article/details/125977540)

#### `nn.AdaptiveAvgPool2d` 函数

##### 参考网页

#### `*参数` 的作用

`*参数` 可以解压参数

```py
a = (0,1,2,3,4,5,6,7,8,9)
b = [0,1,2,3,4,5,6,7,8,9]
print(*a)
print(*b)
```

将 List 和 Tuple 中的元素逐一解压出来

```py
0 1 2 3 4 5 6 7 8 9
0 1 2 3 4 5 6 7 8 9
```

##### 参考网页

TEDxPY - [Python *args 用法笔记](https://blog.csdn.net/weixin_40796925/article/details/107574267)

#### `pip install` 默认安装在 `base` 环境

使用 `pip install` 时改用如下指令即可安装到当前虚拟环境中

```py
python -m pip install **
```

##### 参考网页

timertimer - [在conda虚拟环境中用pip安装包总是在base环境中的解决办法](https://blog.csdn.net/timertimer/article/details/122808662)

### CIFAR10 特化模型

| layer_name | out_size | out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Input | 32\*32 | 3 | None | None | None |
| Conv1 | 32\*32 | 16 | 1 | 1 | 31 |
| Conv2_x | 32\*32 | 16 | 64*4=256 | - | - | - |
| Conv3_x | 16\*16 | 32 | 128*4=512 | - | - | - |
| Conv4_x | 8\*8 | 64 | 256*4=1024 | - | - | - |
| Avgpool | 1\*1 | 64 | None | None | None |
| Flatten | 64 | 1 | None | None | None |
| FC | 10 | 1 | None | None | None |

其中 Conv2_x 、 Conv3_x  、Conv4_x 层由 Block 组成

| layer_name | ResNet_CIFAR10 |
| :---: | :---: |
| Conv2_x | Block\*n |
| Conv3_x | Block\*n |
| Conv4_x | Block\*n |

Block 结构如下

| layer_name | in_size | out_size |  out_channel | kernel_size | stride | padding |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv1 | x\*x | (x/stride)\*(x/stride) | out_channel | 3 | stride | 1 |
| Conv2 | x'\*x' | x'\*x' | out_channel | 3 | 1 | 1 |
| identity |

当基础结构是 Conv3_x 、Conv4_x 、 Conv5_x 的第一层时， `stride=2` 且 identity 为下采样后的输入

```py
nn.Sequential(
    nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
    nn.BatchNorm2d(out_channel * block.expansion),
)
```

当基础结构是 Conv3_x 、Conv4_x 、 Conv5_x 的其他层或在 Conv2_x 层时， `stride=1` 且 identity 为输入本身

```py
nn.Sequential()
```

#### 模型实现

```py
class Block(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # transforming (batch_size * x * x * input_channel) to (batch_size * x * x * output_channel)
        #                                                   or (batch_size * x/2 * x/2 * output_channel)
        # floor(((x - 3 + 2 * 1) / stride) + 1) => floor(x) stride = 1
        #                                       => floor(x/2) stride = 2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x' * x' * output_channel) to (batch_size * x' * x' * output_channel)
        # floor(((x' - 3 + 2 * 1) / 1) + 1) => floor(x')
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
```

```py
class ResNet(nn.Module):
    def __init__(self, block: Block, num_block: List[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_channel = 16
        # transforming (batch_size * 32 * 32 * input_channel) to (batch_size * 32 * 32 * 16)
        # floor(((32 - 3 + 2 * 1) / 1) + 1) => floor(112.5) => floor(112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # transforming (batch_size * 32 * 32 * 16) to (batch_size * 32 * 32 * 16)
        self.conv2_x = self._make_layer(block, 16, num_block, stride=1)
        # transforming (batch_size * 32 * 32 * 16) to (batch_size * 16 * 16 * 32)
        self.conv3_x = self._make_layer(block, 32, num_block, stride=2)
        # transforming (batch_size * 16 * 16 * 16) to (batch_size * 8 * 8 * 64)
        self.conv4_x = self._make_layer(block, 64, num_block, stride=2)
        # transforming (batch_size * 8 * 8 * 64) to (batch_size * 1 * 1 * 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # transforming (batch_size * 64) to (batch_size * num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Block, out_channel: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

```py
def ResNet20() -> ResNet:
    return ResNet(Block, 3)


def ResNet32() -> ResNet:
    return ResNet(Block, 5)


def ResNet44() -> ResNet:
    return ResNet(Block, 7)


def ResNet56() -> ResNet:
    return ResNet(Block, 9)


def ResNet110() -> ResNet:
    return ResNet(Block, 18)


def ResNet1202() -> ResNet:
    return ResNet(Block, 200)
```

## 模型训练

模型训练内容与 AlexNet_CIFAR10 项目相似，相同之处不再赘述

### 封装自定义 Python 库

此次实验中将 AlexNet_CIFAR10 项目中计算数据集均值和方差封装在 `utils` 文件夹下

需要在 `utils` 文件夹下生成空的 `__init__.py` 文件，声明 `utils` 文件夹为封装好的 Python 库

