---
title: PyTorch 实现分类问题总结-续
mathjax: true
categories:
  - - 技术
    - Python
tags:
  - Python
  - PyTorch
  - CNN
description: 针对上一次 Titanic 数据二分类预测实验的拓展和延伸
abbrlink: 49b59d4e
date: 2022-09-12 19:58:44
---

## 问题回顾

### 多线程加载 Dataset

```py
dataset = TitanicDataset("./dataset/titanic/processed_train.csv")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
```

其中参数 num_works 表示载入数据时使用的进程数，此时如果参数的值不为0而使用多进程时会出现报错

> RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase. This probably means that you are not using fork to start your child processes and you have forgotten to use the proper idiom in the main module: if `__name__ == '__main__':` freeze_support() ... The "`freeze_support()`" line can be omitted if the program is not going to be frozen to produce an executable.

#### 解决方法

当参数 num_works 不为 0 时，需要在数据调用前之前加上 `if __name__ == '__main__':`

#### 参考网页

diandianti - [pytorch-Dataloader多进程使用出错](https://www.jianshu.com/p/4a1a92f0efd9)

### K 折交叉校验法

K 折交叉校验法是将训练集划分为 K 份子数据集，然后进行 K 轮训练，在第 K 轮训练时，将第 K 份子数据集作为验证集来评估训练结果，最终会得到 K 个结构相同，参数不同的网络，然后再从中选择评价最好模型

在实际训练中可以将下列代码嵌入训练过程中

```py
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    train_subsampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler)
    # training epoch
```

使得在每一折训练时，训练集和验证集能够变化

#### 参考网页

Jennie_J - [sklearn KFold()](https://blog.csdn.net/weixin_43685844/article/details/88635492)

外部逍遥 - [python sklearn中KFold与StratifiedKFold](https://zhuanlan.zhihu.com/p/150446294)

christianversloot - [How to use K-fold Cross Validation with PyTorch?](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md)

### 归一化问题

#### 归一化与划分训练集和验证集顺序问题

应当先进行训练集划分再进行归一化

因为归一化实际上是从数据集中提取信息，而算法不应知道来自训练集以外信息，否则一定程度上会导致标签泄漏，造成训练结果出现偏差

#### 参考网页

PigOrz - [【关于归一化与反归一化数据统一的问题】：训练集与测试集必须使用同一参数的归一化与反归一化](https://blog.csdn.net/qq_33731081/article/details/103852478)

lizju - [DataFrame的归一化](https://blog.csdn.net/weixin_42227482/article/details/105829627)

Shian150629 - [标准化，归一化与训练-测试集数据处理](https://blog.csdn.net/weixin_43759518/article/details/113880715)

### 模型保存

#### 保存/加载模型参数

1. 保存

    ```py
    torch.save(model.state_dict(), PATH)
    ```

2. 加载

    ```py
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    ```

#### 保存/加载整个模型

1. 保存

    ```py
    torch.save(model, PATH)
    ```

2. 加载

    ```py
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()
    ```

#### 参考网页

爱不持久 - [Pytorch如何保存和加载模型参数](https://blog.csdn.net/wacebb/article/details/108021921)

机器学习入坑者 - [一文梳理pytorch保存和重载模型参数攻略](https://zhuanlan.zhihu.com/p/94971100)

PyTorch - [SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
