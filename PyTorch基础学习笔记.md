### PyTorch基础学习笔记

> PyTorch是美国互联网巨头Facebook在深度学习框架Torch的基础上使用Python重写的一个全新的深度学习框架，它更像NumPy的替代产物，不仅继承了NumPy的众多优点，还支持GPUs计算，在计算效率上要比NumPy有更明显的优势；不仅如此，PyTorch还有许多高级功能，比如拥有丰富的API，可以快速完成深度神经网络模型的搭建和训练。

#### 一、Tensor（张量）

> Tensor在PyTorch中负责存储基本数据，PyTorch针对Tensor也提供了丰富的函数和方法，所以PyTorch中的Tensor与NumPy的数组具有极高的相似性。Tensor是一种高级的API，我们在使用Tensor时并不用了解PyTorch中的高层次架构，也不用明白什么是深度学习、什么是后向传播、如何对模型进行优化、什么是计算图等技术细节。更重要的是，在PyTorch中定义的Tensor数据类型的变量还可以在GPUs上进行运算，而且只需对变量做一些简单的类型转换就能够轻易实现。

##### 1.  Tensor的数据类型

###### （1）torch.FloatTensor：用于生成数据类型为浮点型的Tensor，传递给torch.FloatTensor的参数可以是一个列表，也可以是一个维度值。

```
import torch

a = torch.FloatTensor(2, 3)
b = torch.FloatTensor([2, 3, 4, 5])

print(a)
print(b)
```

在运行后，输出的内容如下：

```
tensor([[6.3369e-10, 2.5038e-12, 4.0046e-11],
        [6.8997e-07, 1.1704e-19, 1.3563e-19]])
tensor([2., 3., 4., 5.])
```

打印输出的两组变量数据类型都显示为浮点型，不同的是，前面的一组是按照我们指定的维度随机生成的浮点型Tensor，而另外一组是按我们给定的列表生成的浮点型Tensor。

###### （2）torch.IntTensor：用于生成数据类型为整型的Tensor，传递给torch.IntTensor的参数可以是一个列表，也可以是一个维度值。

###### （3）torch.rand：用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.rand生成随机数的方法类似，随机生成的浮点数据在0～1区间均匀分布。

###### （4）torch.randn：用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.randn生成随机数的方法类似，随机生成的浮点数的取值满足均值为0、方差为1的正太分布。

###### （5）torch.range：用于生成数据类型为浮点型且自定义起始范围和结束范围的Tensor，所以传递给torch.range的参数有三个，分别是范围的起始值、范围的结束值和步长，其中，步长用于指定从起始值到结束值的每步的数据间隔。

PS：运行出现：torch.range is deprecated in favor of torch.arange and will be removed in 0.5. Note that arange generates values in [start; end), not [start; end].

改成torch.arange就好，但是注意区间会变。

    import torch
    a =torch.arange(1,20,1)
    print(a)
在运行后，输出的内容如下：

```
tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19])
```

###### （6）torch.zeros：用于生成数据类型为浮点型且维度指定的Tensor，不过这个浮点型的Tensor中的元素值全部为0。

##### 2.  Tensor的运算

###### （1）torch.abs：将参数传递到torch.abs后返回输入参数的绝对值作为输出，输入参数必须是一个Tensor数据类型的变量。

###### （2）torch.add：将参数传递到torch.add后返回输入参数的求和结果作为输出，输入参数既可以全部是Tensor数据类型的变量，也可以一个是Tensor数据类型的变量，另一个是标量。

无论是调用torch.add对两个Tensor数据类型的变量进行计算，还是完成Tensor数据类型的变量和标量的计算，计算方式都和NumPy中的数组的加法运算如出一辙。

###### （3）torch.clamp：对输入参数按照自定义的范围进行裁剪，最后将参数裁剪的结果作为输出。

所以输入参数一共有三个，**分别是需要进行裁剪的Tensor数据类型的变量、裁剪的上边界和裁剪的下边界**，具体的裁剪过程是：*使用变量中的每个元素分别和裁剪的上边界及裁剪的下边界的值进行比较，如果元素的值小于裁剪的下边界的值，该元素就被重写成裁剪的下边界的值；同理，如果元素的值大于裁剪的上边界的值，该元素就被重写成裁剪的上边界的值。*

```
import torch

a = torch.randn(2, 3)
print(a)

b = torch.clamp(a, -0.1, 0.1)
print(b)
```

在运行后，输出的内容如下：

```
tensor([[-0.0423, -0.1678, -0.3539],
        [-0.4776,  0.2040, -0.8480]])
tensor([[-0.0423, -0.1000, -0.1000],
        [-0.1000,  0.1000, -0.1000]])
```

###### （4）torch.div：将参数传递到torch.div后返回输入参数的求商结果作为输出，同样，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合。

###### （5）torch.mul：将参数传递到torch.mul后返回输入参数求积的结果作为输出，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合。

###### （6）torch.pow：将参数传递到torch.pow后返回输入参数的求幂结果作为输出，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合。

###### （7）torch.mm：将参数传递到torch.mm后返回输入参数的求积结果作为输出，不过这个求积的方式和之前的torch.mul运算方式不太一样。

**torch.mm运用矩阵之间的乘法规则进行计算**，被传入的参数会被当作矩阵进行处理，参数的维度自然也要满足矩阵乘法的前提条件，即前一个矩阵的行数必须和后一个矩阵的列数相等，否则不能进行计算。

###### （8）torch.mv：将参数传递到torch.mv后返回输入参数的求积结果作为输出。

**torch.mv运用矩阵与向量之间的乘法规则进行计算**，被传入的参数中的第1个参数代表矩阵，第2个参数代表向量，顺序不能颠倒。

#### 二、搭建一个简易神经网络

```
import torch  # 导入包

# 定义四个整形变量
batch_n = 100  # batch_n是在一个批次中输入数据的数量，值是100，这意味着我们在一个批次中输入100个数据
hidden_layer = 100  # hidden_layer用于定义经过隐藏层后保留的数据特征的个数，这里有100个，因为我们的模型只考虑一层隐藏层，所以在代码中仅定义了一个隐藏层的参数；
input_data = 1000  # 每个数据包含的数据特征有input_data个，因为input_data的值是1000，所以每个数据的数据特征就是1000个
output_data = 10  # output_data是输出的数据，值是10，我们可以将输出的数据看作一个分类结果值的数量，个数10表示我们最后要得到10个分类结果值。

"一个批次的数据从输入到输出的完整过程是："
"  先输入100个具有1000个特征的数据，经过隐藏层后变成100个具有100个特征的数据，"
"  再经过输出层后输出100个具有10个分类结果值的数据，在得到输出结果之后计算损失并进行后向传播，"
"  这样一次模型的训练就完成了，然后循环这个流程就可以完成指定次数的训练，并达到优化模型参数的目的"

x = torch.randn(batch_n, input_data)  # 输入层维度为（100,1000)
y = torch.randn(batch_n, output_data)  # 输出层维度为（100,10）

"由于我们现在并没有好的权重参数的初始化方法，所以选择通过torch.randn来生成指定维度的随机参数作为其初始化参数，尽管这并不是一个好主意"

w1 = torch.randn(input_data, hidden_layer)  # 从输入层到隐藏层的权重参数维度为（1000,100）
w2 = torch.randn(hidden_layer, output_data)  # 从隐藏层到输出层的权重参数维度为（100,10）

"权重参数的维度是怎么定义下来的呢? 其实，只要我们把整个过程看作矩阵连续的乘法运算，就自然能够很快明白了"
"（100,1000）X（1000,100）=（100，100）  、（100,100)X(100,10)=(100,10)"

# 接下来会使用梯度下降的方法来优化神经网络的参数，所以必须定义后向传播的次数和梯度下降使用的学习速率。
epoch_n = 20  # 训练的总次数
learning_rate = 1e-6  # 学习速率 learning_rate的值为1e-6，表示1×10-6，即0.000001

# 通过循环的方式让程序进行20次训练，来完成对初始化权重参数的优化和调整。
# 循环内的是神经网络模型具体的前向传播和后向传播代码，参数的优化和更新使用梯度下降来完成
for epoch in range(epoch_n):
    "在这个神经网络的前向传播中，通过两个连续的矩阵乘法计算出预测结果，在计算的过程中还对矩阵乘积的结果使用clamp方法进行裁剪，"
    "将小于零的值全部重新赋值为0，这就像加上了一个ReLU激活函数的功能。前向传播得到的预测结果通过y_pred来表示."
    h1 = x.mm(w1)  # 100＊1000
    h1 = h1.clamp(min=0)
    y_pred = h1.mm(w2)  # 100＊10

    "在得到了预测值后就可以使用预测值和真实值来计算误差值了。我们用loss来表示误差值，对误差值的计算使用了均方误差函数。"
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss))

    "之后的代码部分就是通过实现后向传播来对权重参数进行优化了，为了计算方便，我们的代码实现使用的是每个节点的链式求导结果，"
    "在通过计算之后，就能够得到每个权重参数对应的梯度分别是grad_w1和grad_w2。"
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)   # tensor.t()是实现转置

    grad_h = grad_y_pred.clone()       # tensor.clone()返回一个张量的副本，其与原张量的尺寸和数据类型相同
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)

    "在得到参数的梯度值之后，按照之前定义好的学习速率对w1和w2的权重参数进行更新，在代码中每次训练时，我们都会对loss的值进行打印输出"
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

在运行后，输出的内容如下：

```
Epoch:0, Loss:49054628.0000
Epoch:1, Loss:94610104.0000
Epoch:2, Loss:301032448.0000
Epoch:3, Loss:542303616.0000
Epoch:4, Loss:94321856.0000
Epoch:5, Loss:14120633.0000
Epoch:6, Loss:6071595.5000
Epoch:7, Loss:3444106.7500
Epoch:8, Loss:2400539.0000
Epoch:9, Loss:1919369.3750
Epoch:10, Loss:1656904.8750
Epoch:11, Loss:1486809.1250
Epoch:12, Loss:1359978.6250
Epoch:13, Loss:1256496.5000
Epoch:14, Loss:1167784.3750
Epoch:15, Loss:1089764.7500
Epoch:16, Loss:1020094.2500
Epoch:17, Loss:957429.0625
Epoch:18, Loss:900682.0625
Epoch:19, Loss:849140.9375
```

loss值从之前的巨大误差逐渐缩减，这说明我们的模型经过20次训练和权重参数优化之后，得到的预测的值和真实值之间的差距越来越小了。

#### 三、自动梯度（torch.autograd）

> 对于深度的神经网络模型的前向传播使用简单的代码就能实现，但是很难实现涉及该模型中后向传播梯度计算部分的代码，其中最困难的就是对模型计算逻辑的梳理。
>
> 通过使用torch.autograd包，可以使模型参数自动计算在优化过程中需要用到的梯度值，在很大程度上帮助降低了实现后向传播代码的复杂度。

torch.autograd包的主要功能是完成神经网络后向传播中的链式求导，手动实现链式求导的代码会给我们带来很大的困扰，而torch.autograd包中丰富的类减少了这些不必要的麻烦。**实现自动梯度功能的过程大致为：先通过输入的Tensor数据类型的变量在神经网络的前向传播过程中生成一张计算图，然后根据这个计算图和输出结果准确计算出每个参数需要更新的梯度，并通过完成后向传播完成对参数的梯度更新。**

在实践中完成自动梯度需要用到torch.autograd包中的Variable类对我们定义的Tensor数据类型变量进行封装，在封装后，计算图中的各个节点就是一个Variable对象，这样才能应用自动梯度的功能。

如果已经按照如上方式完成了相关操作，则在选中了计算图中的某个节点时，这个节点必定会是一个Variable对象，用X来代表我们选中的节点，那么X.data代表Tensor数据类型的变量，X.grad也是一个Variable对象，不过它表示的是X的梯度，在想访问梯度值时需要使用X.grad.data。

我们下面也搭建一个二层结构的神经网络模型，便于和上面对比：

```
import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)
"Variable(torch.randn(batch_n, input_data), requires_grad=False) 这段代码就是之前讲到的用Variable类对Tensor数据类型变量进行封装的操作。"
"在以上代码中还使用了一个requires_grad参数，这个参数的赋值类型是布尔型，如果requires_grad的值是False，那么表示该变量在进行自动梯度计算的过程中不会保留梯度值。"
"我们将输入的数据x和输出的数据y的requires_grad参数均设置为False，这是因为这两个变量并不是我们的模型需要优化的参数，而两个权重w1和w2的requires_grad参数的值为True。"

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data))

    loss.backward()

    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

```

*之前代码中的后向传播计算部分变成了新代码中的loss.backward()，这个函数的功能在于让模型根据计算图自动计算每个节点的梯度值并根据需求进行保留，有了这一步，我们的权重参数w1.data和w2.data就可以直接使用在自动梯度过程中求得的梯度值w1.data.grad和w2.data.grad，并结合学习速率来对现有的参数进行更新、优化了。*

*在代码的最后还要将本次计算得到的各个参数节点的梯度值通过grad.data.zero_()全部置零，如果不置零，则计算的梯度值会被一直累加，这样就会影响到后续的计算。*

在运行后，输出的内容如下：

```
Epoch:0, Loss:47338116.0000
Epoch:1, Loss:97949408.0000
Epoch:2, Loss:374966656.0000
Epoch:3, Loss:746421376.0000
Epoch:4, Loss:56399544.0000
Epoch:5, Loss:20339994.0000
Epoch:6, Loss:11709585.0000
Epoch:7, Loss:7755887.0000
Epoch:8, Loss:5540422.5000
Epoch:9, Loss:4160533.2500
Epoch:10, Loss:3238873.2500
Epoch:11, Loss:2592523.7500
Epoch:12, Loss:2122478.2500
Epoch:13, Loss:1771840.0000
Epoch:14, Loss:1504529.2500
Epoch:15, Loss:1296778.1250
Epoch:16, Loss:1132661.2500
Epoch:17, Loss:1001148.3125
Epoch:18, Loss:894226.1250
Epoch:19, Loss:806185.3125
```

从结果来看，对参数的优化在顺利进行，因为loss值也越来越低了。

#### 四、自定义传播函数

> 除了可以采用自动梯度方法，我们还可以通过构建一个继承了torch.nn.Module的新类，来完成对前向传播函数和后向传播函数的重写。在这个新类中，我们使用forward作为前向传播函数的关键字，使用backward作为后向传播函数的关键字。

```
import torch
from torch.autograd import Variable

batch_n = 64
hidden_layer = 100
input_data = 1000
output_data = 10

"首先通过classModel(torch.nn. Module)完成了类继承的操作，之后分别是类的初始化，以及forward函数和backward函数。"
"forward函数实现了模型的前向传播中的矩阵运算，backward实现了模型的后向传播中的自动梯度计算，"
"后向传播如果没有特别的需求，则在一般情况下不用进行调整。在定义好类之后，我们就可以对其进行调用了"

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, w1, w2):
        x = torch.mm(input, w1)
        x = torch.clamp(x, min=0)
        x = torch.mm(x, w2)
        return x

    def backward(self):
        pass
        return


model = Model()

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)

epoch_n = 30
learning_rate = 1e-6

for epoch in range(epoch_n):
    # 模型通过“y_pred = model(x, w1, w2)”来完成对模型预测值的输出，并且整个训练部分的代码被简化了
    y_pred = model(x, w1, w2)

    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data))
    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
```

在运行后，输出的内容如下：

```
Epoch:0, Loss:28151866.0000
Epoch:1, Loss:24739038.0000
Epoch:2, Loss:25950180.0000
Epoch:3, Loss:27835844.0000
Epoch:4, Loss:27276116.0000
Epoch:5, Loss:22496930.0000
Epoch:6, Loss:15407659.0000
Epoch:7, Loss:8983852.0000
Epoch:8, Loss:4875907.5000
Epoch:9, Loss:2690179.0000
Epoch:10, Loss:1615441.5000
Epoch:11, Loss:1082076.7500
Epoch:12, Loss:798210.6250
Epoch:13, Loss:630277.8750
Epoch:14, Loss:519263.3438
Epoch:15, Loss:438640.0938
Epoch:16, Loss:376149.4688
Epoch:17, Loss:325793.0312
Epoch:18, Loss:284095.4375
Epoch:19, Loss:249010.8906
Epoch:20, Loss:219174.5469
Epoch:21, Loss:193616.3125
Epoch:22, Loss:171656.0625
Epoch:23, Loss:152690.8281
Epoch:24, Loss:136200.7969
Epoch:25, Loss:121802.2422
Epoch:26, Loss:109181.9062
Epoch:27, Loss:98082.3125
Epoch:28, Loss:88291.6172
Epoch:29, Loss:79632.8047
```

从结果来看，对参数的优化同样在顺利进行，每次输出的loss值也在逐渐减小

Write by sheen