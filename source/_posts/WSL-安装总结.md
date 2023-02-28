---
title: WSL 安装总结
mathjax: true
categories:
  - - 技术
    - Linux
tags:
  - WSL
  - Ubuntu
description: 配置 WSL 在 Ubuntu 上进行训练
abbrlink: f1f0ebac
date: 2022-08-24 09:38:36
---

考虑到虚拟机不能使用 GPU 加速，现在打算配置 WSL 在 Ubuntu 上进行训练

## 环境要求

Windows 10 版本 2004 及更高版本

## WSL 安装步骤

1. 启用系统功能

    控制面板 =⇒ 程序 =⇒ 程序和功能 =⇒ 启用或关闭 windows 功能，勾选下列选项
    - Hyper-V
    - 适用于 Linux 的 Windows 子系统
    - 虚拟机平台

    然后重新启动计算机

2. 下载并安装[适用于 x64 计算机的 WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

3. 将 WSL 2 设置为默认版本

    打开 PowerShell ，输入如下指令

    ```PowerShell
    wsl --set-default-version 2
    ```

4. 安装 Linux 分发

    在 PowerShell 中输入如下指令可以查看可获取的 Linux 分发版本

    ```PowerShell
    wsl --list --online
    ```

    推荐选择 Ubuntu-20.04.3 LTS

    输入如下指令安装 Linux 分发

    ```PowerShell
    wsl --install -d <Distribution Name>
    ```

    > WSL --install 默认安装固定版本的 Ubuntu
    > -d 参数可以选择任意其他 Linux 分发版本
    > <Distribution Name\> 替换为 Linux 分发版本的名称

5. 安装后设置用户账号和密码

### WSL 安装参考网页

Microsoft - [旧版 WSL 的手动安装步骤](https://docs.microsoft.com/zh-cn/windows/wsl/install-manual)

憨憨不敢_ - [2022 window下安装ubuntu22.04（wsl升级 包含 podman & docker ）](https://blog.csdn.net/weixin_45191709/article/details/125871102)

## WSL 配置

1. WSL2 连接 Windows 防火墙

    1. 检查 WSL 与 Windows 连接

        在 PowerShell 中输入如下指令获取 Windows 本机 IP 和 WSL IP

        ```PowerShell
        ipconfig
        ```

        Windows 本机 IP 地址如下

        ```PowerShell
        无线局域网适配器 WLAN:

            IPv4 地址 . . . . . . . . . . . . :
        ```

        WSL IP 地址如下

        ```PowerShell
        以太网适配器 vEthernet (WSL):

            IPv4 地址 . . . . . . . . . . . . :
        ```

        分别在 PowerShell 和 WSL 中输入下列指令检查连接

        ```PowerShell
        ping <IPv4 Address>
        ```

        > Linux 系统中 Ping 指令不会自行停止，需要使用`Ctrl + C`停止指令

    2. 若不能 ping 通，则检查 Windows 防火墙：

        控制面板 =⇒ 系统和安全 =⇒ Windows Defender 防火墙 =⇒ 高级设置

        - 入站规则
        - 出站规则

        启用其中所有 WSL 规则

    3. 若不存在该规则则需要自行创建：在 PowerShell 中以管理员身份输入如下指令

        ```PowerShell
        New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -InterfaceAlias "vEthernet (WSL)"  -Action Allow
        New-NetFirewallRule -DisplayName "WSL" -Direction Outbound  -InterfaceAlias "vEthernet (WSL)"  -Action Allow
        ```

2. 配置 Ubuntu apt 源

    备份原 apt 源

    ```sh
    sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak
    ```

    编辑 apt 源

    ```sh
    sudo vi /etc/apt/sources.list
    ```

    添加国内源

      - 清华源

        ```sh
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name> main restricted universe multiverse
        deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name> main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-updates main restricted universe multiverse
        deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-updates main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-backports main restricted universe multiverse
        deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-backports main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-security main restricted universe multiverse
        deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ <Ubuntu Distribution Name>-security main restricted universe multiverse
        ```

      - 中科大源

        ```sh
        deb https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name> main restricted universe multiverse
        deb-src https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name> main restricted universe multiverse
        deb https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-security main restricted universe multiverse
        deb-src https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-security main restricted universe multiverse
        deb https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-updates main restricted universe multiverse
        deb-src https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-updates main restricted universe multiverse
        deb https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-backports main restricted universe multiverse
        deb-src https://mirrors.ustc.edu.cn/ubuntu/ <Ubuntu Distribution Name>-backports main restricted universe multiverse
        ```

    更新源

    ```sh
    sudo apt-get update
    ```

    更新软件

    ```sh
    sudo apt-get upgrade
    ```

### WSL / Ubuntu 配置问题

1. Ubuntu 切换阿里云源后提示缺少公钥

    解决方法：

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32

### WSL / Ubuntu 配置参考网页

清华大学开源软件镜像站 - [Ubuntu 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)
中国科技大学开源软件镜像站 - [Ubuntu 源使用帮助](https://mirrors.ustc.edu.cn/help/ubuntu.html)
weixin_43858295 - [Ubuntu 换阿里云源后更新提示： GPG error 缺少公钥解决方法](https://blog.csdn.net/weixin_43858295/article/details/123959824)
