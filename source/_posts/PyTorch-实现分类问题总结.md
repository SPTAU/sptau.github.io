---
title: PyTorch 实现分类问题总结
mathjax: true
categories:
  - - 技术
    - Python
tags:
  - Python
  - CNN
  - PyTorch
description: 使用 Titanic Dataset 作为数据集，利用 PyTorch 实现乘客是否生还的二分类预测
abbrlink: 4fd12987
date: 2022-08-28 11:37:14
---

## 数据集处理

### 实验记录

在 Kaggle 网站上下载 [Titanic 数据集](https://www.kaggle.com/competitions/titanic/data?select=train.csv)，解压在 `./dataset/Titanic` 下

由于 Jupiter 可以实现数据的实时可视化，在此使用 Jupiter 进行数据集的观察与处理

1. 新建 `.ipynb` 文件后导入 Python 库

    ```py
    import os
    import pandas as pd
    ```

2. 设置数据集的相对地址

    ```py
    TRAIN_PATH = "./dataset/titanic/train.csv"
    PROCESSED_TRAIN_PATH = "./dataset/titanic/processed_train.csv"
    ```

3. 读取并显示 train dataset 信息

    ```py
    train_data = pd.read_csv(TRAIN_PATH)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  891 non-null    int64
     1   Survived     891 non-null    int64
     2   Pclass       891 non-null    int64
     3   Name         891 non-null    object
     4   Sex          891 non-null    object
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64
     7   Parch        891 non-null    int64
     8   Ticket       891 non-null    object
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object
     11  Embarked     889 non-null    object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    ```

4. 可以看到 PassengerId 、 Name 、 Ticket 三列数据与是否生存无关，将其丢弃

    ```py
    train_data = train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    object
     3   Age       714 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Cabin     204 non-null    object
     8   Embarked  889 non-null    object
    dtypes: float64(2), int64(4), object(3)
    memory usage: 62.8+ KB
    ```

5. Age 数据缺失，现在使用均值进行补充

    ```py
    avg_age = train_data['Age'].mean()
    train_data['Age'] = train_data['Age'].fillna(avg_age)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    object
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Cabin     204 non-null    object
     8   Embarked  889 non-null    object
    dtypes: float64(2), int64(4), object(3)
    memory usage: 62.8+ KB
    ```

6. 由于 Cabin 数据确实太多，将其丢弃

    ```py
    train_data = train_data.drop(['Cabin'], axis=1)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    object
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Embarked  889 non-null    object
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB
    ```

7. 查看 Embarked 数据中的众数

    ```py
    train_data['Embarked'].value_counts()
    ```

    ```py
    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64
    ```

    使用众数进行补充缺失的数据

    ```py
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    object
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Embarked  891 non-null    object
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB
    ```

8. 查看 Sex 数据的情况

    ```py
    train_data['Sex'].head()
    ```

    ```py
    0      male
    1    female
    2    female
    3    female
    4      male
    Name: Sex, dtype: object
    ```

    将 Sex 数据的值映射成数值

    ```py
    sex_2_dict = {"male": 0, "female":1}
    train_data['Sex'] = train_data['Sex'].map(sex_2_dict)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    int64
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Embarked  891 non-null    object
    dtypes: float64(2), int64(4), object(1)
    memory usage: 55.8+ KB
    ```

9. 查看 Embarked 数据的情况

    ```py
    train_data['Embarked'].head()
    ```

    ```py
    0    S
    1    C
    2    S
    3    S
    4    S
    Name: Embarked, dtype: object
    ```

    将 Embarked 数据的值映射成数值

    ```py
    embarked_2_dict = {"C": 0, "Q": 1, "S": 2}
    train_data['Embarked'] = train_data['Embarked'].map(embarked_2_dict)
    train_data.info()
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   Survived  891 non-null    int64
     1   Pclass    891 non-null    int64
     2   Sex       891 non-null    int64
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64
     5   Parch     891 non-null    int64
     6   Fare      891 non-null    float64
     7   Embarked  891 non-null    int64
    dtypes: float64(2), int64(6)
    memory usage: 55.8 KB
    ```

10. 将处理好的数据集导出到文件夹中

    ```py
    train_data.to_csv(PROCESSED_TRAIN_PATH, index=False)
    ```

### 回顾

#### conda 安装 Jupiter 报错

打开 clash 后，用 conda 安装 Jupiter 报错

```bash
CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/current_repodata.json>
```

##### 错误出现原因

clash 代理影响 conda 与 conda 源的连接

##### 解决方法1

在用户文件夹下的 `.condarc` 文件中添加代理端口

clash 的默认代理端口为 7890

```txt
proxy_servers:
http: http://127.0.0.1:7890
https: https://127.0.0.1:7890
```

但在后续的安装中依旧报错

```txt
Retrieving notices: ...working... failed
ssl.SSLEOFError: EOF occurred in violation of protocol (_ssl.c:1129)
```

此解决方法**无效**

##### 解决方法2

关闭 clash

##### 参考网页

wielice - [conda 配置本机代理](https://blog.csdn.net/weixin_38789153/article/details/121598812)

好事要发生 - [An HTTP error occurred when trying to retrieve this URL. HTTP errors are often intermittent......](https://blog.csdn.net/weixin_43505418/article/details/123711162)

#### `pandas.read_csv()` 函数

从 CSV 文件中读取数据到 DataFrame

```py
pandas.read_csv(filepath_or_buffer, sep=NoDefault.no_default,
delimiter=None, header='infer', names=NoDefault.no_default,
index_col=None, usecols=None, squeeze=None, prefix=NoDefault.no_default,
mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
true_values=None, false_values=None, skipinitialspace=False,
skiprows=None, skipfooter=0, nrows=None, na_values=None,
keep_default_na=True, na_filter=True, verbose=False,
skip_blank_lines=True, parse_dates=None, infer_datetime_format=False,
keep_date_col=False, date_parser=None, dayfirst=False,
cache_dates=True, iterator=False, chunksize=None, compression='infer',
thousands=None, decimal='.', lineterminator=None, quotechar='"',
quoting=0, doublequote=True, escapechar=None, comment=None,
encoding=None, encoding_errors='strict', dialect=None,
error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None,
delim_whitespace=False, low_memory=True, memory_map=False,
float_precision=None, storage_options=None)
```

##### 部分重要参数解析

1. filepath_or_buffer

    str ，路径对象或类文件对象

    任何有效的字符串路径都可以接受

    该字符串可以是一个 URL

    有效的 URL 方案包括 http 、 ftp 、 s3 、 gs 和 file

2. sep

    str ，默认为','

    长于 1 个字符且与' \s+ '不同的分隔符将被解释为正则表达式

3. delimiter

    str ，与 sep 参数功能相同，但默认为 'None'

    当 sep 参数与 delimiter 参数均不为 None 时，弹出错误信息

    ```py
    ValueError: Specified a sep and a delimiter; you can only specify one.
    ```

4. header

    int 或 int 列表 或 None ，但默认为 'Infer'

    指定表头在数据中的行数

    - 当 header 参数为默认的 'Infer' 且 names 参数为 None 时，等价于 header = 0 ，此时将读取数据的第一行作为表头/列名
    - 当 header 参数为默认的 'Infer' 且 names 参数不为 None 时，等价于 header = None ，此时将读取 names 参数作为表头/列名
    - 当 header 参数为 int 时，将从整数对应的行号读取列名
    - 当 header 参数为 int 列表时，将从列表对应的行号读取列名
    - 当 skip_blank_lines 参数为 True 且 names 参数不为 None 时，这个参数会忽略注释行和空行，会从**数据的第一行**而不是文件的第一行开始计算行数并读取表头

5. names

    array-like ，可选

    以传入的参数作为表头

    - 参数中不允许有重复的内容
    - 当 header 参数为 int 或 int 列表且 names 参数不为 None 时，将从 header 参数对应的行号读取列名，然后根据 names 参数对表头进行替换

6. dtype

    Type name 或 column → type 字典，可选，默认为 None

    将数据以参数数据类型读取

7. skiprows

    list-like 或 int 或 可调用对象 ，可选，默认为 None

    - 当 skiprows 的参数为 int 或 列表 时，将跳过对应行号的数据，对余下数据进行读取

    - 当 skiprows 的参数为 可调用对象（如函数） 时，函数会先遍历行索引，进行条件判断，然后跳过函数返回值为 True 的行号的数据，对余下数据进行读取

8. index_col

    int 或 str 或 int / str 序列，可选，默认为 None

    - 当 index_col 的参数为 int 或 str 时，将对应列号/列名用作数据的索引

    - 当 index_col 的参数为 int / str 序列时，将使用 MultiIndex，将对应列号/列名用作数据的索引

##### 参考网页

pandas - [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

用 Python 学机器学习 - [一文掌握 read_csv 函数](https://baijiahao.baidu.com/s?id=1696564586951432790&wfr=spider&for=pc)

#### `pandas.DataFrame.to_csv()` 函数

将 DataFrame 写入 CSV 文件中

```py
DataFrame.to_csv(path_or_buf=None, sep=',', na_rep='',
float_format=None, columns=None, header=True, index=True,
index_label=None, mode='w', encoding=None, compression='infer',
quoting=None, quotechar='"', line_terminator=None, chunksize=None,
date_format=None, doublequote=True, escapechar=None, decimal='.',
errors='strict', storage_options=None)
```

##### 部分重要参数

1. path_or_buf

    str 或 path object 或 file-like object 或 None ，默认为 None

    - 当 path_or_buf 的参数不为 None 时，将把 DataFrame 写入对应路径/文件名中
    - 当 path_or_buf 的参数为 None 时，将把 DataFrame 以字符串形式打印出来
    如果传递的是一个非二进制文件对象，应该用newline=''打开，禁用通用换行。
    如果传递的是二进制文件对象，模式可能需要包含一个'b'。

2. sep

    str ，默认为 ‘,’

    用于输出文件的字段分隔符。

    sep 的参数要求长度为1的字符串

3. na_rep

    str ，默认为 ‘ ’

    用于转换 DataFrame 中的 NaN

4. header

    bool or str 列表，默认为 True

    - 当 header 的参数为 bool 时，将根据 header 的参数决定是否将数据的表头写入文件中

    - 当 header 的参数为 str 列表 时，将把 header 的参数作为表头写入文件中。

5. index

    bool ，默认为 True

    用于决定是否将索引写入文件中

6. mode

    str ，默认为 'w'

    用于设置 Python 的写入模式

##### 参考网页

pandas - [pandas.DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv)

AI阿聪 - [pd.read_csv() 和 pd.to_csv() 常用参数](https://blog.csdn.net/weixin_40431584/article/details/105065464)

quantLearner - [pd.read_csv()||pd.to_csv() 索引问题 index](https://blog.csdn.net/The_Time_Runner/article/details/88353161)

暴躁的猴子 - [pd.to_csv详解](https://blog.csdn.net/orangefly0214/article/details/80764569)

## 模型训练

按照[课程](https://www.bilibili.com/video/BV1Y7411d7Ys)将代码分为四部分

在此之前导入库

```py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
```

### Prepare dataset

构建 Dataset ，完成 Dataloader

```py
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath, sep=",", dtype="float32")
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy.iloc[:, 1:].values)
        self.y_data = torch.tensor(xy.iloc[:, 0].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = TitanicDataset("./dataset/titanic/processed_train.csv")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
```

### Design model using Class

从 torch.nn.Module 继承构建模型

```py
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = nn.Linear(7, 22)
        self.Linear2 = nn.Linear(22, 11)
        self.Linear3 = nn.Linear(11, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
        x = x.squeeze(-1)
        return x


model = Model()
```

### Construct loss and optimizer

使用 PyTorch API 指定 Loss 函数和优化器

```py
criterion = nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Training cycle

完成前向传播，反向传播，更新权值

```py
acc_list = []
for epoch in range(300):

    acc = 0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data                   # 载入数据

        label_pred = model(inputs)              # 前向传播，计算梯度
        loss = criterion(label_pred, labels)    # 计算 Loss

        for j in range(len(label_pred)):        # 计算 acc
            if round(label_pred[i - 1].item()) == labels[i - 1].item():
                acc += 1

        optimizer.zero_grad()                   # 梯度清零
        loss.backward()                         # 反向传播

        optimizer.step()                        # 更新权值

    acc /= len(dataset)
    acc_list.append(acc)                        # 存储 acc
    print("epoch:", epoch, " acc:", acc)

fig, ax = plt.subplots()                        # 绘制 acc 变化曲线
ax.plot(np.arange(300), acc_list)
plt.show()

```

### 回顾

#### 继承 Dataset 类初始化报错

```py
TypeError: object.__new__() takes exactly one argument (the type to instantiate)
```

##### 错误出现原因

发现是将子类的初始化函数名称写成了 `_init_` 而非 `__init__`

子类继承时未重构初始化函数，在创建实例时不能进行初始化

##### 解决方法

修改为 `__init__` 后错误信息消失

##### 参考网页

摩天仑 - [Python 学习坑 —— init](https://blog.csdn.net/weixin_57064740/article/details/121589595)

#### 读取数据时将索引作为数据读入

使用 `pandas.read_csv()` 函数读取数据时，将索引作为数据读入数据中的第一列

##### 错误出现原因

在将处理后的数据写入 CSV 文件时将索引也一并写入，但从 CSV 文件中读取数据时未声明数据中包含有索引

##### 解决方法1

将 DataFrame 数据写入 CSV 文件时使用 `pandas.DataFrame.to_csv(PATH, index=False)`

##### 解决方法2

从 CSV 文件读取数据时使用 `pandas.read_csv(PATH, index_col=0)`

##### 参考网页

hellocsz - [read_csv 文件读写参数详解](https://blog.csdn.net/hellocsz/article/details/79623142)

#### 从 DataFrame 中提取数据报错

代码

```py
self.x_data = torch.from_numpy(xy[:, 1:-1])
self.y_data = torch.from_numpy(xy[:, [0]])
```

错误信息

```py
TypeError: '(slice(None, None, None), slice(1, -1, None))' is an invalid key
```

##### 错误出现原因

DataFrame 的切片操作仅支持 `.loc` 和 `.iloc` 函数

##### 解决方法

使用 `pandas.DataFrame.iloc()` 函数

```py
self.x_data = torch.tensor(xy.iloc[:, 1:])
self.y_data = torch.tensor(xy.iloc[:, 0])
```

`pandas.DataFrame.iloc()` 中的范围是**左开右闭**的

##### 参考网页

pandas - [pandas.DataFrame.iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)

方如一 - [iloc[ ]函数（Pandas库）](https://blog.csdn.net/Fwuyi/article/details/123127754)

#### 从 dataset 读取数据时报错

代码

```py
dataset = TitanicDataset("./dataset/titanic/processed_train.csv")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
```

错误信息

```py
RuntimeError: DataLoader worker (pid(s) 32708, 31720) exited unexpectedly
```

##### 错误出现原因

未知，据推测是多线程工作的支持不好，或者是多线程被某些特定程序杀掉了

##### 解决方法

将 num_workers 参数设置为 0 （默认为 0 ）

```py
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
```

##### 参考网页

xiuxiuxiuxiul - [Pytorch设置多线程进行dataloader时影响GPU运行](https://blog.csdn.net/xiuxiuxiuxiul/article/details/86233500)

#### 计算 Loss 时报错

模型代码

```py
class Model(nn.Module):
    def __init__(self):
        # ......

    def forward(self, x):
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
        return x
```

计算 Loss 代码

```py
for epoch in range(100):
    acc = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        label_pred = model(inputs)
        loss = criterion(label_pred, labels)
        # ......
```

错误信息

```py
ValueError: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])) is deprecated. Please ensure they have the same size.
```

##### 错误出现原因

模型输出结果 `label_pred` 为 [32, 1] ，而 `labels` 为 [32] ，二者的维度不匹配

##### 解决方法

在前馈计算中加入 `pandas.DataFrame.squeeze(-1)` 将输出结果的最后一维压缩

```py
class Model(nn.Module):
    def __init__(self):
        # ......

    def forward(self, x):
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
+       x = x.squeeze(-1)
        return x
```

##### 参考网页

发梦公主 - [深度学习过程的Python报错收集](https://blog.csdn.net/lyh2240465046/article/details/123695335)

pandas - [pandas.DataFrame.squeeze](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.squeeze.html)

### 实验结果

```py
# ......
epoch: 290  acc: 0.6767676767676768
epoch: 291  acc: 0.712682379349046
epoch: 292  acc: 0.7485970819304153
epoch: 293  acc: 0.7542087542087542
epoch: 294  acc: 0.7845117845117845
epoch: 295  acc: 0.6767676767676768
epoch: 296  acc: 0.7485970819304153
epoch: 297  acc: 0.8204264870931538
epoch: 298  acc: 0.8204264870931538
epoch: 299  acc: 0.7901234567901234
```

#### 结果评价

![image.png](https://s2.loli.net/2022/08/28/XserVa7zuvLwYUQ.png)

可以看到训练存在一定的效果，但并不明显，抖动明显

#### 存在的问题

1. 由于 loss 设置问题，不能直观显示 loss 随训练轮数变化

2. 设计模型设计不够完善

3. 本次实验中选取的特征只针对训练集，没有考虑测试集中数据的情况

    在查看测试集中的数据后发现其中的 Fare 和 Age 数据缺失

    可以将训练集和测试集中存在缺失数据的特征都去除

    可以用训练集中的均值、众数填补测试集中的缺失项，但是这样将影响测试集，导致分类结果产生偏移

4. 数据集中特征较少

#### 后续改进

使用 K 折交叉校验法

重新设计训练模块

### 参考网页

刘二大人 - [《 PyTorch 深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys?p=8)
