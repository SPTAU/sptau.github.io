---
title: PointNet++ 论文总结
mathjax: true
categories:
  - - 技术
    - 学习
tags:
  - PointNet++
  - 点云
description: 'PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space'
abbrlink: 3e81cdbb
date: 2022-08-24 09:46:55
---

## 第一遍阅读论文

### 感受

1. [PointNet++](https://proceedings.neurips.cc/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html) 的网络结构和 FPN 结构很像，都使用“由上至下——下采样”、“横向连接——跨级链接”及“由下至上——上采样”的结构，从而对多尺度特征图进行融合，将高层的语义信息与低层的几何细节结合。
2. 相比起 [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html) 论文，这篇论文更难懂一点。或许是因为 PointNet 结构很简单。

### 疑惑

1. PointNet++ 对点云进行多次降采样后调用 PointNet

   那么调用的是 PointNet 的哪一个模型？是分类模型还是语义分割模型？ PointNet 的输出结果是怎样的？

2. 采样层使用最远点采样法进行分组

    计算距离的坐标基点是哪里？是原点吗？计算的是什么距离？是欧氏距离吗？

## 第二遍阅读论文

### Abstract

1. PointNet 的缺点

    PointNet 不能很好地捕捉到局部结构信息，其识别细腻度模式的能力较弱，且难以应用在复杂场景

2. PointNet++ 的优点

    PointNet++ 引入分层神经网络，在嵌套分区上递归应用 PointNet ；通过距离学习不同尺度的局部信息；加入集合学习层，以自适应地结合来自多个尺度的特征

### Introduction

1. PointNet++ 的特征提取过程

   1. PointNet++ 通过底层空间的距离度量将点集划分为若干重叠的局部单元
   2. PointNet++ 从局部单元中提取捕捉精细几何结构的局部特征；
   3. PointNet++ 将所有局部特征被进一步分组为更大的单元并被处理以产生更高层次的特征
   4. PointNet++ 重复步骤 3 ，直到我们得到整个点集的特征。

2. PointNet++ 面临的问题

   1. 如何产生局部单元

        通过最远点采样法选取分区质心

   2. 如何通过局部特征学习器抽象出点集或局部特征

   3. 如何在密度不均匀的点集进行自适应分组

        - 在图像中，较小的卷积核可以提升 CNN 提取特征的能力
        - 在点集中，在较小的邻域不足以让网络稳健地提取特征

### Problem Statement

设计一个输入为点集，输出为分级的语义信息的函数

### Method

![PointNet++ 网络结构图](https://s2.loli.net/2022/08/19/xhqnjzpP1SsQKrR.png)

1. 回顾 PointNet

    $$f(x_1,...,x_n) ≈ \gamma(\max_{i=1,...,n}\{h(x_i)\})\tag{1}$$

    PointNet 简单高效，具有很强的鲁棒性，但是缺乏在不同尺度上捕捉局部结构的能力

2. 分层特征学习

    PointNet++ 的分层结构由多层 set abstraction 层级构成

    每层 set abstraction 层级由采样层、分组层和 PointNet 层构成

    - 采样层从点集中选择一些点作为中心点

    - 分组层寻找中心点周围的相邻点来构建局部单元

    - PointNet 层使用 mini-PointNet 从局部单元中提取特征

    set abstraction 层级的输入是 $N \times (d + C)$ 的矩阵，输出是 $N' \times (d' + C')$ 的矩阵

    > 输入矩阵来自 $N $ 个具有 $ d $ 维坐标和 $ C$ 维特征的点
    > 输出矩阵由 $N' $ 个具有 $ d' $ 维坐标和 $ C'$ 维特征的点组成

    1. 采样层

        该层的输入是点集 $\{x_1, x_2, ..., x_n \}$ ，输出是子点集 $\{x_{i_1}, x_{i_2}, ..., x_{i_n} \}$

        采样层使用最远采样法，从点集中选择几何距离较远的点作为子点集，子点集中的点即为中心点

        常用的几何距离有欧氏距离和测地距离

        选择初始点的方法有随机选取和选取距离点云重心的最远点

        最远采样法相比于随机采样，能够更好覆盖整个点集；相比于 CNN ，最远采样法以一种依赖数据的方式生成感受野

    2. 分组层

        该层的输入是整个点集构成的 $N \times (d + C)$ 矩阵和采样层输出的子点集的坐标构成的 $N' \times d$ 矩阵，输出是 $N' \times K \times (d + C)$ 的矩阵

        > K 是中心点附近的点数，可以自适应调整

        分组层使用球状半径查询，从点集中寻找中心点半径范围内的 K 个点

        球状半径查询相比于 kNN 搜索，能够保证固定的区域尺度

    3. PointNet 层

        该层的输入是 $N' \times K \times (d + C)$ 矩阵，输出是 $N' \times (d + C')$ 的矩阵

3. 非均匀采样密度下的特征学习

    1. 多尺度分组

        对每个尺度的分组都送入 PointNet 层进行特征提取，然后将特征连接起来作为中心点的特征

        计算成本很高

    2. 多分辨率分组

        将大尺度的特征和送入 PointNet 层提取的小尺度特征进行拼接

        在密度较低的区域，自适应提升大尺度特征的权重

        在密度较高的区域，自适应提升送入 PointNet 层提取的小尺度特征的权重

    3. 随机丢弃输入数据

        在将数据输入进 PointNet 层之前进行 95% 的重新采样

4. 分割任务中点特征的传播

    在语义分割任务中，希望得到所有原始点的点特征

    1. 将所有的点都作为中心点进行采样分组

        计算成本很高

    2. 将特征用内插特征值从子采样点传播到原始点

        内插特征值使用基于 k 近邻的反距离加权平均

        将内插的特征值与跳过的链接点特征相连接

        $${f^{(i)}} = {\frac {\sum_{i=1}^{k} {w_{i}(x)} {f}_{i}^{(j)}} {\sum_{i=1}^{k}{w_{i}(x)}}} \quad where \quad {w}_{i}(x) = {\frac {1} {d(x, x_i)^p}}, j = 1, ..., C \qquad \tag{2}$$

### Experiments

1. 数据集

    1. MNIST

        70k 张二维图片，按 6:1 比例划分训练集和测试集

    2. ModelNet40

        40 类 12311 个 CAD 模型，按 4:1 比例划分训练集和测试集

    3. SHREC15

        50 类 1200 个 CAD 模型，每个类别包括 24 种形状，使用五折交叉验证

    4. ScanNet

        1513 个室内扫描重建场景，按 4:1 比例划分训练集和测试集

2. 欧氏几何空间中的点集分类

    在 MNIST 和 ModelNet40 数据集中的表现明显优于 PointNet

    在模拟的不均匀数据集中，保持很好的性能

3. 场景语义分割

    对抽样密度变化具有稳健性

4. 非欧几何空间中的点集分类

    对每个模型构建由成对的测地线距离诱导的度量空间

    提取 WKS 、 HKS 和多尺度高斯曲率等特征作为特征，而不是以三维坐标作为特征

    因为以三维坐标作为特征不能揭示内在结构，而且受姿势变化的影响很大

5. 特征可视化

    通过特征可视化可以看出 PointNet++ 学习到了物体的基本结构

### conclusion

改进方向：

如何通过在每个局部区域分享更多的计算来加快网络的推理速度

## 复现结果

代码源自 yanx27 - [PyTorch Implementation of PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

输入如下指令开始训练 PointNet++ 分类模型

```py
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
```

训练最终结果

```py
Epoch 200 (200/200):
Train Instance Accuracy: 0.976626
Test Instance Accuracy: 0.920146, Class Accuracy: 0.890121
Best Instance Accuracy: 0.926133, Class Accuracy: 0.898150
End of training...
```

输入如下指令开始测试 PointNet++ 分类模型

```py
python test_classification.py --log_dir pointnet2_cls_ssg
```

测试最终结果

```py
Test Instance Accuracy: 0.927751, Class Accuracy: 0.896012
```

## 参考文章

刘昕宸 - [搞懂 PointNet++ ，这篇文章就够了！](https://zhuanlan.zhihu.com/p/266324173)

Guoguang Du - [最远点采样(Farthest Point Sampling)介绍](https://blog.csdn.net/dsoftware/article/details/107184116)

## 论文总结

### PointNet++ 针对 PointNet 提出的改进

1. PointNet 只是提取点云中各个点的空间特征，然后聚合成全局信息。这样不能很好地提取局部结构信息

    PointNet++ 通过采样和分组将点集划分为不同的重叠区域，即整合局部邻域

2. PointNet 在聚合成全局信息时只使用了最大值池化。这样会损失大量信息

    PointNet++ 采用多层神经网络，对点集中的分组不断进行下采样，提取不同尺度下的 local-global feature

3. PointNet 中的语义分割模型提取的中间特征是全局特征和局部特征拼接而成，特征的差异度不够

    PointNet++ 先进行下采样再进行上采样，使用跨级链接和内插特征值拼接对应层的 local-global feature
