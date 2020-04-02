# 安装学到的教训：

# ※安装完anaconda后要先切换镜像源

切换镜像源的命令：

1.添加镜像：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
2.切换镜像（设置搜索时显示通道地址）：

conda config --set show_channel_urls yes

3.查看镜像源（用来确认）：

conda config --show channels

# 1.安装前的准备：先安装显卡驱动+CUDA+CUDNN，然后安装anaconda，注意勾选添加到环境变量

# 2.安装：先创建环境，然后激活环境，再去pytorch.org查找安装命令，再运行命令即可安装成功

##### 1)创建环境：conda create -n pytorch_gpu pip python=3.7

# 2)激活和关闭pytorch_gpu环境的命令

###### To activate this environment, use

conda activate pytorch_gpu

###### To deactivate an active environment, use

 conda deactivate

3)我的安装命令

```英语
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

# 3.将Pytorch设置到PyCharm中去

file--setting--Project Interpreter--设置按钮--add--conda environment--existing environment--inetrpreter那里找到anaconda安装目录的envs，再找到之前安装环境的那个名字的文件夹（pytorch_gpu）找到下面的python.exe

# 4.测试安装是否成功

```英语
import torch
import torchvision

#显示PyTorch的版本
print("hello PyTorch! 您的版本是：{}".format(torch.__version__))
#测试CUDA是否可用
print("CUDA的可用状态：")
print(torch.cuda.is_available()) #true即为可用
```

# 5.参考资料：

https://blog.csdn.net/SpadgerZ/article/details/89468756

https://blog.csdn.net/baidu_26646129/article/details/88380598



Write by sheen