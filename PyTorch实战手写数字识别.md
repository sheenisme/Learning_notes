### PyTorch实战手写数字识别

> 具体过程是：先使用已经提供的训练数据对搭建好的神经网络模型进行训练并完成参数优化；然后使用优化好的模型对测试数据进行预测，对比预测值和真实值之间的损失值，同时计算出结果预测的准确率。另外，在将要搭建的模型中会用到卷积神经网络模型。

#### 一、 torch和torchvision

在PyTorch中有两个核心的包，分别是torch和torchvision。torchvision包的主要功能是实现数据的处理、导入和预览等，所以如果需要对计算机视觉的相关问题进行处理，就可以借用在torchvision包中提供的大量的类来完成相应的工作。

```
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
```

首先，导入必要的包。对这个手写数字识别问题的解决只用到了torchvision中的部分功能，所以这里通过from torchvision import方法导入其中的两个子包datasets和transforms，我们将会用到这两个包。

之后，我们就要想办法获取手写数字的训练集和测试集。使用torchvision.datasets可以轻易实现对这些数据集的训练集和测试集的下载，只需要使用torchvision.datasets再加上需要下载的数据集的名称就可以了，比如在这个问题中我们要用到手写数字数据集，它的名称是MNIST，那么实现下载的代码就是torchvision.datasets.MNIST。**其他常用的数据集如COCO、ImageNet、CIFCAR等都可以通过这个方法快速下载和载入**。实现数据集下载的代码如下：

```
# 实现数据集下载
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)
data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

```

其中，**root用于指定数据集在下载之后的存放路径**，这里存放在根目录下的data文件夹中；**transform用于指定导入数据集时需要对数据进行哪种变换操作**，在后面会介绍详细的变换操作类型，注意，要提前定义这些变换操作；**train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分**。

#### 二、PyTorch之torch.transforms

> 我们知道，在计算机视觉中处理的数据集有很大一部分是图片类型的，而在PyTorch中实际进行计算的是Tensor数据类型的变量，所以我们首先需要解决的是数据类型转换的问题，如果获取的数据是格式或者大小不一的图片，则还需要进行归一化和大小缩放等操作，庆幸的是，这些方法在torch.transforms中都能找到。在torch.transforms中提供了丰富的类对载入的数据进行变换，现在让我们看看如何进行变换。

在torch.transforms中有大量的数据变换类，其中有很大一部分可以用于实现数据增强（Data Argumentation）。**若在我们需要解决的问题上能够参与到模型训练中的图片数据非常有限，则这时就要通过对有限的图片数据进行各种变换，来生成新的训练集了，这些变换可以是缩小或者放大图片的大小、对图片进行水平或者垂直翻转等，都是数据增强的方法。**不过在手写数字识别的问题上可以不使用数据增强的方法，因为可用于模型训练的数据已经足够了。对数据进行载入及有相应变化的代码如下：

```
# 需要提前定义的对载入的数据进行变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1307], std=[0.3081])])

```

我们可以将以上代码中的**torchvision.transforms.Compose类看作一种容器**，它能够同时对多种数据变换进行组合。**传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作**。

在以上代码中，**在torchvision.transforms.Compose中只使用了一个类型的转换变换transforms.ToTensor和一个数据标准化变换transforms.Normalize**。这里使用的*标准化变换也叫作标准差变换法，这种方法需要使用原始数据的均值（Mean）和标准差（StandardDeviation）来进行数据的标准化，在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布*。计算公式如下：

![img](E:\Github socures\Learning_notes\epub_23914630_212-1586067253572.jpg)

不过我们在这里偷了一个懒，均值和标准差的值并非来自原始数据的，而是自行定义了一个，不过仍然能够达到我们的目的。

##### 1.  在torchvision.transforms中常用的数据变换操作。

###### （1）torchvision.transforms.Resize：用于对载入的图片数据按我们需求的大小进行缩放。

传递给这个类的参数可以是一个整型数据，也可以是一个类似于（h,w）的序列，其中，h代表高度，w代表宽度，但是如果使用的是一个整型数据，那么表示缩放的宽度和高度都是这个整型数据的值。

###### （2）torchvision.transforms.Scale：用于对载入的图片数据按我们需求的大小进行缩放，用法和torchvision.transforms.Resize类似。

###### （3）torchvision.transforms.CenterCrop：用于对载入的图片以图片中心为参考点，按我们需要的大小进行裁剪。

传递给这个类的参数可以是一个整型数据，也可以是一个类似于（h,w）的序列。

###### （4）torchvision.transforms.RandomCrop：用于对载入的图片按我们需要的大小进行随机裁剪。

传递给这个类的参数可以是一个整型数据，也可以是一个类似于（h,w）的序列。

###### （5）torchvision.transforms.RandomHorizontalFlip：用于对载入的图片按随机概率进行水平翻转。

我们可以通过传递给这个类的参数自定义随机概率，如果没有定义，则使用默认的概率值0.5。

###### （6）torchvision.transforms.RandomVerticalFlip：用于对载入的图片按随机概率进行垂直翻转。

我们可以通过传递给这个类的参数自定义随机概率，如果没有定义，则使用默认的概率值0.5。

###### （7）torchvision.transforms.ToTensor：用于对载入的图片数据进行类型转换，将之前构成PIL图片的数据转换成Tensor数据类型的变量，让PyTorch能够对其进行计算和处理。

###### （8）torchvision.transforms.ToPILImage：用于将Tensor变量的数据转换成PIL图片数据，主要是为了方便图片内容的显示。

#### 三、数据预览和数据装载

> 在数据下载完成并且载入后，我们还需要对数据进行装载。我们可以将数据的载入理解为对图片的处理，在处理完成后，我们就需要将这些图片打包好送给我们的模型进行训练了，而装载就是这个打包的过程。

在装载时**通过batch_size的值来确认每个包的大小**，**通过shuffle的值来确认是否在装载的过程中打乱图片的顺序**。装载图片的代码如下：

```
 # 对数据进行装载(处理）
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)
```

**对数据的装载使用的是torch.utils.data.DataLoader类，类中的dataset参数用于指定我们载入的数据集名称，batch_size参数设置了每个包中的图片数据个数，代码中的值是64，所以在每个包中会包含64张图片。将shuffle参数设置为True，在装载的过程会将数据随机打乱顺序并进行打包。**

在装载完成后，我们可以选取其中一个批次的数据进行预览。进行数据预览的代码如下：

```
# 进行数据预览
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)
```

在以上代码中**使用了iter和next来获取一个批次的图片数据和其对应的图片标签，然后使用torchvision.utils中的make_grid类方法将一个批次的图片构造成网格模式**。

需要传递给torchvision.utils.make_grid的参数就是一个批次的装载数据，每个批次的装载数据都是4维的，**维度的构成从前往后分别为batch_size、channel、height和weight，分别对应一个批次中的数据个数、每张图片的色彩通道数、每张图片的高度和宽度**。

*在通过torchvision.utils.make_grid之后，图片的维度变成了（channel,height, weight），这个批次的图片全部被整合到了一起，所以在这个维度中对应的值也和之前不一样了，但是色彩通道数保持不变。若我们想使用Matplotlib将数据显示成正常的图片形式，则使用的数据首先必须是数组，其次这个数组的维度必须是（height, weight,channel），即色彩通道数在最后面。所以我们要通过numpy和transpose完成**原始数据类型的转换和数据维度的交换**，这样才能够使用Matplotlib绘制出正确的图像。*

在完成数据预览的代码中，我们先打印输出了这个批次中的数据的全部标签，然后才对这个批次中的所有图片数据进行显示，代码如下：

```
[tensor(5), tensor(3), tensor(9), tensor(5), tensor(0), tensor(0), tensor(4), tensor(4), tensor(3), tensor(3), tensor(3), tensor(0), tensor(7), tensor(3), tensor(3), tensor(8), tensor(1), tensor(1), tensor(4), tensor(9), tensor(4), tensor(5), tensor(3), tensor(5), tensor(7), tensor(0), tensor(4), tensor(7), tensor(2), tensor(8), tensor(7), tensor(5), tensor(4), tensor(7), tensor(9), tensor(7), tensor(2), tensor(6), tensor(9), tensor(9), tensor(6), tensor(2), tensor(5), tensor(8), tensor(5), tensor(6), tensor(0), tensor(1), tensor(4), tensor(8), tensor(0), tensor(0), tensor(7), tensor(5), tensor(5), tensor(9), tensor(1), tensor(7), tensor(0), tensor(4), tensor(3), tensor(2), tensor(7), tensor(7)]
```

可以看到，打印输出的首先是64张图片对应的标签，然后是64张图片的预览结果:

![Figure_8](E:\Github socures\Learning_notes\Figure_8.png)

#### 四、模型搭建和参数优化

> 在顺利完成数据装载后，我们就可以开始编写卷积神经网络模型的搭建和参数优化的代码了。

因为我们想要**搭建一个包含了卷积层、激活函数、池化层、全连接层的卷积神经网络来解决这个问题**，所以模型在结构上会和之前简单的神经网络有所区别，当然，各个部分的功能实现依然是通过torch.nn中的类来完成的，比如卷积层使用torch.nn.Conv2d类方法来搭建；激活层使用torch.nn.ReLU类方法来搭建；池化层使用torch.nn.MaxPool2d类方法来搭建；全连接层使用torch.nn.Linear类方法来搭建。

```
# 搭建卷积神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(-1, 14 * 14 * 128)
            x = self.dense(x)
            return x

```

因为这个问题并不复杂，所以我们选择搭建一个在结构层次上有所简化的卷积神经网络模型，在结构上使用了两个卷积层：一个最大池化层和两个全连接层，下面对其具体的使用方法进行补充说明：

###### （1）torch.nn.Conv2d：用于搭建卷积神经网络的卷积层，主要的输入参数有输入通道数、输出通道数、卷积核大小、卷积核移动步长和Paddingde值。

其中，输入通道数的数据类型是整型，用于确定输入数据的层数；输出通道数的数据类型也是整型，用于确定输出数据的层数；卷积核大小的数据类型是整型，用于确定卷积核的大小；卷积核移动步长的数据类型是整型，用于确定卷积核每次滑动的步长；Paddingde的数据类型是整型，值为0时表示不进行边界像素的填充，如果值大于0，那么增加数字所对应的边界像素层数。

###### （2）torch.nn.MaxPool2d：用于实现卷积神经网络中的最大池化层，主要的输入参数是池化窗口大小、池化窗口移动步长和Paddingde值。

同样，池化窗口大小的数据类型是整型，用于确定池化窗口的大小。池化窗口步长的数据类型也是整型，用于确定池化窗口每次移动的步长。Paddingde值和在torch.nn.Conv2d中定义的Paddingde值的用法和意义是一样的。

###### （3）torch.nn.Dropout: torch.nn.Dropout类用于防止卷积神经网络在训练的过程中发生过拟合。

其工作原理简单来说就是在模型训练的过程中，以一定的随机概率将卷积神经网络模型的部分参数归零，以达到减少相邻两层神经连接的目的。下图显示了Dropout方法的效果：

![image-20200407112822849](E:\Github socures\Learning_notes\image-20200407112822849.png)

在图中打叉的神经节点就是被随机抽中并丢弃的神经连接，正是因为选取方式的随机性，所以在模型的每轮训练中选择丢弃的神经连接也是不同的，**这样做是为了让我们最后训练出来的模型对各部分的权重参数不产生过度依赖，从而防止过拟合。**对于torch.nn.Dropout类，我们可以对随机概率值的大小进行设置，如果不做任何设置，就使用默认的概率值0.5。

前向传播forward函数中的内容：首先，**经过self.conv1进行卷积处理**；然后**进行x.view(-1, 14*14*128)，对参数实现扁平化**，因为之后紧接着的就是全连接层，所以如果不进行扁平化，则全连接层的实际输出的参数维度和其定义输入的维度将不匹配，程序会报错；最后，**通过self.dense定义的全连接进行最后的分类**。

> 在编写完搭建卷积神经网络模型的代码后，我们就可以开始对模型进行训练和对参数进行优化了。

首先，定义在训练之前使用哪种损失函数和优化函数：

```
model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

以上代码中**定义了计算损失值的损失函数使用的是交叉熵，也确定了优化函数使用的是Adam自适应优化算法，需要优化的参数是在Model中生成的全部参数，因为没有定义学习速率的值，所以使用默认值；**然后，*通过打印输出的方式查看搭建好的模型的完整结构，只需使用print(model)就可以了*，输出结果如下：

```
Model(
  (conv1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense): Sequential(
    (0): Linear(in_features=25088, out_features=1024, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=1024, out_features=10, bias=True)
  )
)
```

最后，卷积神经网络模型进行模型训练和参数优化的代码如下：

```
# 进行模型训练和参数优化
n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train), 100 * running_correct / len( data_train), 100 * testing_correct / len(data_test)))
```

总的训练次数是5次，训练中的大部分代码和之前相比没有大的改动，增加的内容都在原来的基础上加入了更多的打印输出，其目的是更好地显示模型训练过程中的细节，同时，在每轮训练完成后，会使用测试集验证模型的泛化能力并计算准确率。在模型训练过程中打印输出的结果如下：

    Epoch 0/5
    ----------
    Loss is:0.0019, Train Accuracy is:96.0000%, Test Accuracy is:98.0000
    Epoch 1/5
    ----------
    Loss is:0.0007, Train Accuracy is:98.0000%, Test Accuracy is:98.0000
    Epoch 2/5
    ----------
    Loss is:0.0004, Train Accuracy is:99.0000%, Test Accuracy is:98.0000
    Epoch 3/5
    ----------
    Loss is:0.0003, Train Accuracy is:99.0000%, Test Accuracy is:98.0000
    Epoch 4/5
    ----------
    Loss is:0.0002, Train Accuracy is:99.0000%, Test Accuracy is:98.0000

可以看到，结果表现非常不错，训练集达到的最高准确率为99.73%，而测试集达到的最高准确率为98.96%。如果我们使用功能更强大的卷积神经网络模型，则会取得比现在更好的结果。

为了验证我们训练的模型是不是真的已如结果显示的一样准确，则最好的方法就是随机选取一部分测试集中的图片，用训练好的模型进行预测，看看和真实值有多大的偏差，并对结果进行可视化。测试过程的代码如下：

```
# 用训练好的模型进行预测，看看和真实值有多大的偏差，并对结果进行可视化
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=4, shuffle=True)
X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_, pred = torch.max(pred, 1)

print("Predict Label is:", [i for i in pred.data])
print("Real Label is:", [i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)

# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
mean = [0.1307]
std = [0.3081]
img = img*std+mean
plt.imshow(img)
```

用于测试的数据标签结果输出的结果如下：

```
Predict Label is: [tensor(3), tensor(6), tensor(2), tensor(1)]
Real Label is: [tensor(3), tensor(6), tensor(2), tensor(1)]
```

在输出结果中，第1个结果是我们训练好的模型的预测值，第2个结果是这4个测试数据的真实值。对测试数据进行可视化，如下图所示：

![Figure_10](E:\Github socures\Learning_notes\Figure_10.png)

可以看到，在图中可视化的这部分测试集图片，模型的预测结果和真实的结果是完全一致的。当然，如果想选取更多的测试集进行可视化，则只需将batch_size的值设置得更大。





Write by sheen