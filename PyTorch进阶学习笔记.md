### PyTorch进阶学习笔记

#### 一、PyTorch之torch.nn

> PyTorch中的torch.nn包提供了很多与实现神经网络中的具体功能相关的类，这些类涵盖了深度神经网络模型在搭建和参数优化过程中的常用内容，比如神经网络中的卷积层、池化层、全连接层这类层次构造的方法、防止过拟合的参数归一化方法、Dropout方法，还有激活函数部分的线性激活函数、非线性激活函数相关的方法，等等。

下面使用PyTorch的torch.nn包来简化我们之前的代码，开始部分的代码变化不大，如下所示：

```
    import torch
    from torch.autograd import Variable
    batch_n =100
    hidden_layer =100
    input_data =1000
    output_data =10

    x =Variable(torch.randn(batch_n, input_data), requires_grad =False)
    y =Variable(torch.randn(batch_n, output_data), requires_grad =False)
```

##### 1.神经网络模型

模型搭建的代码如下：

```
    models =torch.nn.Sequential(
        torch.nn.Linear(input_data, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, output_data)
    )
```

torch.nn.Sequential括号内的内容就是我们搭建的神经网络模型的具体结构，这里首先通过torch.nn.Linear(input_data, hidden_layer)完成从输入层到隐藏层的线性变换，然后经过激活函数及torch.nn.Linear(hidden_layer, output_data)完成从隐藏层到输出层的线性变换。下面分别对这三个类进行详细介绍：

###### （1）torch.nn.Sequential: torch.nn.Sequential类是torch.nn中的一种序列容器，通过在容器中嵌套各种实现神经网络中具体功能相关的类，来完成对神经网络模型的搭建，最主要的是，参数会按照我们定义好的序列自动传递下去。

我们可以将嵌套在容器中的各个部分看作各种不同的模块，这些模块可以自由组合。模块的加入一般有两种方式，一种是在以上代码中使用的直接嵌套，另一种是以orderdict有序字典的方式进行传入，这两种方式的唯一区别是，使用后者搭建的模型的每个模块都有我们自定义的名字，而前者默认使用从零开始的数字序列作为每个模块的名字。

使用orderdict有序字典进行传入来搭建的模型代码如下：

```
    hidden_layer =100
    input_data =1000
    output_data =10

    from collections import OrderedDict
    models =torch.nn.Sequential(OrderedDict([
        ("Line1", torch.nn.Linear(input_data, hidden_layer)),
        ("Relu1", torch.nn.ReLU()),
        ("Line2", torch.nn.Linear(hidden_layer, output_data))])
    )
    print(models)
```

这里对该模型的结构进行打印输出，结果如下：

```
Sequential(
  (Line1): Linear(in_features=1000, out_features=100, bias=True)
  (Relu1): ReLU()
  (Line2): Linear(in_features=100, out_features=10, bias=True)
)
```

我们会发现，对模块使用自定义的名称可让我们更便捷地找到模型中相应的模块并进行操作。

###### （2）torch.nn.Linear: torch.nn.Linear类用于定义模型的线性层，即完成前面提到的不同的层之间的线性变换。

torch.nn.Linear类接收的参数有三个，分别是输入特征数、输出特征数和是否使用偏置，**设置是否使用偏置的参数是一个布尔值，默认为True，即使用偏置。**在实际使用的过程中，我们只需将输入的特征数和输出的特征数传递给torch.nn.Linear类，就会自动生成对应维度的权重参数和偏置，对于生成的权重参数和偏置，我们的**模型默认使用了一种比之前的简单随机方式更好的参数初始化方法**。

根据我们搭建模型的输入、输出和层次结构需求，它的输入是在一个批次中包含100个特征数为1000的数据，最后得到100个特征数为10的输出数据，中间需要经过两次线性变换，所以要使用两个线性层，两个线性层的代码分别是torch.nn.Linear(input_data, hidden_layer)和torch.nn.Linear(hidden_layer, output_data)。可看到，其代替了之前使用矩阵乘法方式的实现，代码更精炼、简洁。

###### （3）torch.nn.ReLU: torch.nn.ReLU类属于非线性激活分类，在定义时默认不需要传入参数。

> 当然，在torch.nn包中还有许多非线性激活函数类可供选择，比如之前讲到的PReLU、LeakyReLU、Tanh、Sigmoid、Softmax等。

在掌握torch.nn.Sequential、torch.nn.Linear和torch.nn.RelU的使用方法后，快速搭建更复杂的多层神经网络模型变为可能，而且在整个模型的搭建过程中不需要对在模型中使用到的权重参数和偏置进行任何定义和初始化说明，因为参数已经完成了自动生成。

接下来对已经搭建好的模型进行训练并对参数进行优化，代码如下：

```
    epoch_n =10000
    learning_rate =1e-4
    loss_fn =torch.nn.MSELoss()
```

和之前相比，计算损失函数的代码发生了改变，现在使用的是在torch.nn包中已经定义好的均方误差函数类torch.nn.MSELoss来计算损失值，而之前的代码是根据损失函数的计算公式来编写的。

##### 2.损失函数

下面简单介绍在torch.nn包中常用的损失函数的具体用法，如下所述：

###### （1）torch.nn.MSELoss: torch.nn.MSELoss类使用<u>均方误差函数</u>对损失值进行计算，在定义类的对象时不用传入任何参数，但在使用实例时需要输入两个维度一样的参数方可进行计算。

示例如下：

```
    import torch
    from torch.autograd import  Variable
    loss_f =torch.nn.MSELoss()
    x =Variable(torch.randn(100,100))
    y =Variable(torch.randn(100,100))
    loss =loss_f(x, y)
    print(loss.data)
```

打印输出的结果如下：

```
 tensor(1.9776)
```

以上代码首先通过随机方式生成了两个维度都是（100,100）的参数，然后使用均方误差函数来计算两组参数的损失值。

###### （2）torch.nn.L1Loss: torch.nn.L1Loss类使用<u>平均绝对误差函数</u>对损失值进行计算，同样，在定义类的对象时不用传入任何参数，但在使用实例时需要输入两个维度一样的参数进行计算。

###### （3）torch.nn.CrossEntropyLoss :torch.nn.CrossEntropyLoss类用于<u>计算交叉熵</u>，在定义类的对象时不用传入任何参数，在使用实例时需要输入两个满足交叉熵的计算条件的参数

##### 3.实例

在学会使用PyTorch中的优化函数之后，我们就可以对自己建立的神经网络模型进行训练并对参数进行优化了，代码如下：

```
    for epoch in range(epoch_n):
        y_pred =models(x)
        loss =loss_fn(y_pred, y)
        if epoch%1000 ==0:
          print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data[0]))
        models.zero_grad()

        loss.backward()

        for param in models.parameters():
          param.data  -=param.grad.data*learning_rate
```

以上代码中的绝大部分和之前训练和优化部分的代码是一样的，但是**参数梯度更新的方式发生了改变**。**因为使用了不同的模型搭建方法，所以访问模型中的全部参数是通过对“models.parameters()”进行遍历完成的，然后才对每个遍历的参数进行梯度更新。**其打印输入结果的方式是每完成1000次训练，就打印输出当前的loss值，最后输出的结果如下：

```
Epoch:0, Loss:1.0693
Epoch:1000, Loss:0.9905
Epoch:2000, Loss:0.9219
Epoch:3000, Loss:0.8612
Epoch:4000, Loss:0.8073
Epoch:5000, Loss:0.7585
Epoch:6000, Loss:0.7137
Epoch:7000, Loss:0.6722
Epoch:8000, Loss:0.6335
Epoch:9000, Loss:0.5971
```

从该结果可以看出，参数的优化效果比较理想，loss值被控制在相对较小的范围之内，这和我们增加了训练次数有很大关系。

#### 二、torch.optim

*到目前为止，代码中的神经网络权重的参数优化和更新还没有实现自动化，并且目前使用的优化方法都有固定的学习速率，所以优化函数相对简单，如果我们自己实现一些高级的参数优化算法，则优化函数部分的代码会变得较为复杂。*

在PyTorch的torch.optim包中提供了非常多的可实现参数自动优化的类，比如SGD、AdaGrad、RMSProp、Adam等，这些类都可以被直接调用，使用起来也非常方便。我们使用自动化的优化函数实现方法对之前的代码进行替换，新的代码如下：

```
    import torch
    from torch.autograd import Variable
    batch_n =100
    hidden_layer =100
    input_data =1000
    output_data =10

    x =Variable(torch.randn(batch_n, input_data), requires_grad =False)
    y =Variable(torch.randn(batch_n, output_data), requires_grad=False)

    models =torch.nn.Sequential(
        torch.nn.Linear(input_data, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, output_data)
    )

    epoch_n =10000
    learning_rate =1e-4
    loss_fn =torch.nn.MSELoss()

    optimzer =torch.optim.Adam(models.parameters(), lr =learning_rate)
```

这里使用了torch.optim包中的torch.optim.Adam类作为我们的**模型参数的优化函数**，在**torch.optim.Adam类中输入的是被优化的参数和学习速率的初始值，如果没有输入学习速率的初始值，那么默认使用0.001这个值。因为我们需要优化的是模型中的全部参数，所以传递给torch.optim.Adam类的参数是models.parameters。另外，Adam优化函数还有一个强大的功能，就是可以对梯度更新使用到的学习速率进行自适应调节，所以最后得到的结果自然会比之前的代码更理想。**进行模型训练的代码如下：

```
    for epoch in range(epoch_n):
        y_pred =models(x)
        loss =loss_fn(y_pred, y)
        print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data[0]))
        optimzer.zero_grad()

        loss.backward()
        optimzer.step()
```

在以上代码中有几处代码和之前的训练代码不同，这是因为我们引入了优化算法，所以通过直接调用optimzer.zero_grad来完成对模型参数梯度的归零；并且在以上代码中增加了optimzer.step，它的主要功能是使用计算得到的梯度值对各个节点的参数进行梯度更新。这里只进行20次训练并打印每轮训练的loss值，结果如下：

```
Epoch:0, Loss:1.1326
Epoch:1, Loss:1.1100
Epoch:2, Loss:1.0880
Epoch:3, Loss:1.0665
Epoch:4, Loss:1.0455
Epoch:5, Loss:1.0249
Epoch:6, Loss:1.0048
Epoch:7, Loss:0.9852
Epoch:8, Loss:0.9660
Epoch:9, Loss:0.9472
Epoch:10, Loss:0.9289
Epoch:11, Loss:0.9111
Epoch:12, Loss:0.8937
Epoch:13, Loss:0.8766
Epoch:14, Loss:0.8599
Epoch:15, Loss:0.8435
Epoch:16, Loss:0.8276
Epoch:17, Loss:0.8120
Epoch:18, Loss:0.7967
Epoch:19, Loss:0.7818
```

在看到这个结果后我们会很惊讶，因为使用torch.optim.Adam类进行参数优化后仅仅进行了20次训练，得到的loss值就已经远远低于之前进行6000次优化训练的结果。所以，如果对torch.optim中的优化算法类使用得当，就更能帮助我们优化好模型中的参数。



Write by sheen