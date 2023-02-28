---
title: PyTorch 实现多分类任务总结
mathjax: true
categories:
  - - 技术
    - Python
tags:
  - Python
  - PyTorch
description: 本次试验以 Kaggle 竞赛中的 Otto Group Product Classification Challenge 作为数据集
abbrlink: 7efb9c78
date: 2022-09-12 19:57:15
---

## 数据集处理

### 实验记录

在 Kaggle 网站上下载 [Otto Group Product Classification Challenge 数据集](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data)，解压在 `./dataset/otto-group-product-classification-challenge` 下

1. 导入数据集并查看基本信息

   导入库

    ```py
    import os

    import numpy as np
    import pandas as pd
    ```

   设置 dataset 地址

    ```py
    TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/train.csv"
    TEST_PATH = "./dataset/otto-group-product-classification-challenge/test.csv"
    SAMPLE_SUBMISSION_PATH = "./dataset/otto-group-product-classification-challenge/sampleSubmission.csv"
    PROCESSED_TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/processed_train.csv"
    ```

   读取 training dataset

    ```py
    train_data = pd.read_csv(TRAIN_PATH, index_col=0)
    ```

    显示 training dataset 信息

    ```py
    train_data.info()
    ```

    ```py
    Output exceeds the size limit. Open the full output data in a text editor
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61878 entries, 1 to 61878
    Data columns (total 94 columns):
     #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
     0   feat_1   61878 non-null  int64
     1   feat_2   61878 non-null  int64
     2   feat_3   61878 non-null  int64
    ...
     92  feat_93  61878 non-null  int64
     93  target   61878 non-null  object
    dtypes: int64(93), object(1)
    memory usage: 44.8+ MB
    ```

    查看 training dataset 前几行

    ```py
    train_data.head()
    ```

    ![training dataset(before)](https://s2.loli.net/2022/09/08/3YcZkMEpTrs5eQI.png)

    查找是否存在缺失值

    ```py
    train_data.isnull().sum()
    ```

    ```py
    feat_1     0
    feat_2     0
    feat_3     0
    feat_4     0
    feat_5     0
              ..
    feat_90    0
    feat_91    0
    feat_92    0
    feat_93    0
    target     0
    Length: 94, dtype: int64
    ```

    读取 testing dataset

    ```py
    test_data = pd.read_csv(TEST_PATH, index_col=0)
    ```

    显示 testing dataset 信息

    ```py
    test_data.info()
    ```

    ```py
    Output exceeds the size limit. Open the full output data in a text editor
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 144368 entries, 1 to 144368
    Data columns (total 93 columns):
     #   Column   Non-Null Count   Dtype
    ---  ------   --------------   -----
     0   feat_1   144368 non-null  int64
     1   feat_2   144368 non-null  int64
     2   feat_3   144368 non-null  int64
    ...
     91  feat_92  144368 non-null  int64
     92  feat_93  144368 non-null  int64
    dtypes: int64(93)
    memory usage: 103.5 MB
    ```

    查看 testing dataset 前几行

    ```py
    test_data.head()
    ```

    ![testing dataset(before)](https://s2.loli.net/2022/09/08/JCoyXLOQSTUMe3w.png)

    查找是否存在缺失值

    ```py
    test_data.isnull().sum()
    ```

    ```py
    feat_1     0
    feat_2     0
    feat_3     0
    feat_4     0
    feat_5     0
              ..
    feat_89    0
    feat_90    0
    feat_91    0
    feat_92    0
    feat_93    0
    Length: 93, dtype: int64
    ```

    统计 target 列中的类别和数量

    ```py
    train_data['target'].value_counts()
    ```

    ```py
    Class_2    16122
    Class_6    14135
    Class_8     8464
    Class_3     8004
    Class_9     4955
    Class_7     2839
    Class_5     2739
    Class_4     2691
    Class_1     1929
    Name: target, dtype: int64
    ```

2. 数据处理

    根据上面的信息，可以看到 training dataset 的特征为 int ，标签为 Classs_1 ~ Class_9 的字符串

    现在需要将标签转换为 one-hot 格式

    ```py
    train_data = pd.get_dummies(train_data)
    ```

    确认处理结果

    ```py
    train_data.head()
    ```

    ![training dataset(after)](https://s2.loli.net/2022/09/08/yTz1bLJh6EuU2WV.png)

    ```py
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61878 entries, 1 to 61878
    Columns: 102 entries, feat_1 to target_Class_9
    dtypes: int64(93), uint8(9)
    memory usage: 44.9 MB
    ```

    ```py
    train_data.notnull().sum()
    ```

    ```py
    feat_1            61878
    feat_2            61878
    feat_3            61878
    feat_4            61878
    feat_5            61878
                      ...
    target_Class_5    61878
    target_Class_6    61878
    target_Class_7    61878
    target_Class_8    61878
    target_Class_9    61878
    Length: 102, dtype: int64
    ```

    写入到 CSV 文件中

    ```py
    train_data.to_csv(PROCESSED_TRAIN_PATH, index=False)
    ```

### 回顾

#### `pandas.isnull()` 函数

可以以布尔类型返回各行各列是否存在缺失
加上 sum() 函数可以统计各列的缺失情况

##### 参考网站

若尘公子 - [#有空学04# pandas缺失数据查询](https://zhuanlan.zhihu.com/p/158684561)

pandas - [pandas.isnull](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html)

#### `pandas.value_count()` 函数

可以返回该列中数据种类及其数量

方便后续进行格式转换

##### 参考网站

快乐的皮卡丘呦呦 - [Pandas中查看列中数据的种类及个数](https://www.bbsmax.com/A/mo5k0wkndw/)
pandas - [pandas.DataFrame.value_counts](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.value_counts.html)

#### `pandas.get_dummies()` 函数

`pandas.get_dummies()` 函数会将非数值型数据转换为 One-Hot 格式

在该数据集中即使不指定 columns 参数，也只会转换 target 一列

##### 参考网页

ChaoFeiLi - [操作pandas某一列实现one-hot](https://blog.csdn.net/ChaoFeiLi/article/details/115345237)

pandas - [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

pandas - [pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

## 模型训练

导入库

```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
```

### 准备数据集

```py
# 派生 CSV_Dataset 类
class OGPCCDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath, sep=",", dtype="float32")
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy.iloc[:, :93].values)
        self.y_data = torch.tensor(xy.iloc[:, 93:].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 设置数据集位置
TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/processed_train.csv"
TEST_PATH = "./dataset/otto-group-product-classification-challenge/test.csv"

# 装载数据集
train_dataset = OGPCCDataset(TRAIN_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

### 设计网络模型

```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 9)

    def forward(self, x):
        x = x.view(-1, 93)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


model = Net()
```

### 设计 Loss 和优化器

```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

#### 交叉熵损失函数

$$Loss = -\sum^n_{i=1} y_i \log y'_i$$

交叉熵损失函数通常用于多分类任务的损失函数

##### NLLLoss

NLLLoss 是将 Label 转换为 One-Hot 形式后与输出结果进行交叉熵计算

##### `Torch.nn.CrossEntropyLoss()` 函数

`Torch.nn.CrossEntropyLoss()` 函数是先将输出结果输入到 Softmax 层后取对数，再应用 NLLLoss

即 `Torch.nn.CrossEntropyLoss()` = LogSoftmax + NLLLoss

### 训练过程

```py
def train(epoch):
    epoch_loss_list = []
    epoch_acc_list = []
    for i in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            _, labels = torch.max(targets.data, dim=1)
            total += targets.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / batch_idx
        epoch_acc = 100 * correct / total
        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
        print("[Epoch %3d] loss: %.3f acc: %.3f" % (i + 1, epoch_loss, epoch_acc))
    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_loss_list)
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_acc_list)
    plt.show()


if __name__ == "__main__":
    train(epoch)
```

## 训练结果

Loss 随训练轮次变化

![Loss-Epoch](https://s2.loli.net/2022/09/08/crteKZJRNg1IHMy.png)

acc 随训练轮次变化

![acc-Epoch](https://s2.loli.net/2022/09/08/MVjkvqKGX9Aswy4.png)
