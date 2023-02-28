---
title: PyTorch 搭建 DenseNet 模型对 CIFAR10 数据集分类总结
mathjax: true
categories:
  - - 技术
    - 学习
tags:
  - Python
  - PyTorch
description: >-
  本次实验以 [Densely Connected Convolutional Networks] 作为主要参考来源，使用 PyTorch 搭建
  DenseNet 模型实现对 CIFAR10 数据集的训练和测试
abbrlink: 25308b42
date: 2022-10-07 19:55:00
---

## DenseNet 模型搭建

### 结构分析

#### 通用框架

根据论文中的信息，可以得到常规 DenseNet 模型（忽略 batch_size ）的通用框架如下

| layer_name | out_size | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: |
| Input | 224 \* 224 | None | None | None |
| Conv | 112 \* 112 | 7 | 2 | 3 |
| Maxpool | 56 \* 56 | 3 | 2 | 1 |
| Dense Layer_1 | 56 \* 56 | - | - | - |
| Transition Layer_1 | 28 \* 28 | - | - | - |
| Dense Layer_2 | 28 \* 28 | - | - | - |
| Transition Layer_2 | 14 \* 14 | - | - | - |
| Dense Layer_3 | 14 \* 14 | - | - | - |
| Transition Layer_3 | 7 \* 7 | - | - | - |
| Dense Layer_4 | 7 \* 7 | - | - | - |
| Global Avgpool | 1 \* 1 | 7 | 0 | 0 |
| FC | 1000 | None | None | None |

其中 Dense Layer 由多个 Dense Block 组成

| layer_name | ResNet18 | ResNet34 | ResNet50 |
| :---: | :---: | :---: | :---: |
| Dense Layer_1 | Dense Block \* 6 | Dense Block \* 6 | Dense Block \* 6 |
| Dense Layer_2 | Dense Block \* 12 | Dense Block \* 12 | Dense Block \* 12 |
| Dense Layer_3 | Dense Block \* 24 | Dense Block \* 32 | Dense Block \* 48 |
| Dense Layer_4 | Dense Block \* 16 | Dense Block \* 32 | Dense Block \* 32 |

#### 基础结构

DenseNet 使用的 Dense Block 结构如下

| layer_name | in_size | out_size |  out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Conv1 | x \* x | x \* x | 4 * growth_rate | 1 | 1 | 0 |
| Conv2 | x \* x | x \* x | growth_rate | 3 | 1 | 1 |
| Concatence | None | None | in_channel + growth_rate | None | None | None |

DenseNet 使用的 Dense Block 结构如下

| layer_name | in_size | out_size |  out_channel | kernel_size | stride | padding |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Conv | x \* x | x \* x | in_channel // 2 | 1 | 1 | 0 |
| Avgpool | x \* x | (x/2) \* (x/2) | in_channel // 2 | 2 | 2 | 0 |

#### Conv

DenseNet 的 Conv 层和 ResNet 的 Conv 层不同

ResNet 的 Conv 层实际上是 Conv - BatchNorm - ReLU

而 DenseNet 中除了第一个 Conv 层以外的其他 Conv 层实际上是 ReLU - BatchNorm - Conv

### 网络实现

#### Dense Block

```py
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_features: Tensor) -> Tensor:

        new_features = self.conv1(self.relu1(self.bn1(input_features)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        new_features = torch.cat((input_features, new_features), 1)

        return new_features
```

#### Transition

```py
class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        output = self.Conv(x)
        output = self.AvgPool(output)
        return output
```

#### 通用框架

```py
class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        num_layers: List[int] = [6, 12, 24, 16],
        num_init_features: int = 64,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 112 * 112 * 64)
        # floor(((224 - 7 + 2 * 3) / 2) + 1) => floor(112.5) => floor(112)
        self.Conv = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )
        # transforming (batch_size * 112 * 112 * 64) to (batch_size * 56 * 56 * 64)
        # floor(((112 - 3 + 2 * 1) / 2) + 1) => floor(56.5) => floor(56)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_input_features = num_init_features
        # transforming (batch_size * 56 * 56 * in_channel) to (batch_size * 56 * 56 * (in_channel + num_layers[0] * growth_rate))
        self.DenseBlock1 = self._make_DenseBlock(growth_rate, num_layers[0], num_input_features, drop_rate)
        num_input_features += num_layers[0] * growth_rate

        # transforming (batch_size * 56 * 56 * in_channel) to (batch_size * 28 * 28 * (in_channel // 2))
        self.Transition1 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        # transforming (batch_size * 28 * 28 * in_channel) to (batch_size * 28 * 28 * (in_channel + num_layers[1] * growth_rate))
        self.DenseBlock2 = self._make_DenseBlock(growth_rate, num_layers[1], num_input_features, drop_rate)
        num_input_features += num_layers[1] * growth_rate

        # transforming (batch_size * 28 * 28 * in_channel) to (batch_size * 14 * 14 * (in_channel // 2))
        self.Transition2 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        # transforming (batch_size * 14 * 14 * in_channel) to (batch_size * 14 * 14 * (in_channel + num_layers[2] * growth_rate))
        self.DenseBlock3 = self._make_DenseBlock(growth_rate, num_layers[2], num_input_features, drop_rate)
        num_input_features += num_layers[2] * growth_rate

        # transforming (batch_size * 14 * 14 * in_channel) to (batch_size * 7 * 7 * (in_channel // 2))
        self.Transition3 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        # transforming (batch_size * 7 * 7 * in_channel) to (batch_size * 7 * 7 * (in_channel + num_layers[3] * growth_rate))
        self.DenseBlock4 = self._make_DenseBlock(growth_rate, num_layers[3], num_input_features, drop_rate)
        num_input_features += num_layers[3] * growth_rate

        # transforming (batch_size * 7 * 7 * in_channel) to (batch_size * 1 * 1 * in_channel)
        self.GlobleAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        # transforming (batch_size * 1 * 1 * in_channel) to (batch_size * in_channel)
        self.FC = nn.Linear(num_input_features, num_classes)

    def _make_DenseBlock(self, growth_rate: int, num_layers: int, num_input_features: int, drop_rate: int) -> nn.Sequential:
        layers = []
        for i in range(int(num_layers)):
            layers.append(_DenseLayer(num_input_features, growth_rate, drop_rate=drop_rate))
            num_input_features += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        output = self.Conv(x)
        output = self.MaxPool(output)
        output = self.Transition1(self.DenseBlock1(output))
        output = self.Transition2(self.DenseBlock2(output))
        output = self.Transition3(self.DenseBlock3(output))
        output = self.DenseBlock4(output)
        output = self.GlobleAvgPool(output)
        output = torch.flatten(output, 1)
        output = self.FC(output)
        return output
```

#### 构造网络

```py
def DenseNet121() -> DenseNet:
    return DenseNet(32, [6, 12, 24, 16])


def DenseNet169() -> DenseNet:
    return DenseNet(32, [6, 12, 32, 32])


def DenseNet201() -> DenseNet:
    return DenseNet(32, [6, 12, 48, 32])
```

#### 参考网页

PyTorch - [SOURCE CODE FOR TORCHVISION.MODELS.DENSENET](https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html#densenet121)

Mayurji - [Image-Classification-PyTorch/DenseNet.py](https://github.com/Mayurji/Image-Classification-PyTorch/blob/main/DenseNet.py)

wmpscc - [CNN-Series-Getting-Started-and-PyTorch-Implementation/DenseNet/DenseNet-Torch.py](https://github.com/wmpscc/CNN-Series-Getting-Started-and-PyTorch-Implementation/blob/master/DenseNet/DenseNet-Torch.py)

pytorch - [vision/torchvision/models/densenet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)

## 模型训练

模型训练内容与 AlexNet_CIFAR10 项目相似，相同之处不再赘述

## 总结

DenseNet 和 ResNet 很像， ResNet 是使用了 short cut，而 DenseNet 可以理解为将所有输出都进行 short cut 连接到了其后面的所有输出
