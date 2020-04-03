### PyTorch基础学习笔记

> PyTorch是美国互联网巨头Facebook在深度学习框架Torch的基础上使用Python重写的一个全新的深度学习框架，它更像NumPy的替代产物，不仅继承了NumPy的众多优点，还支持GPUs计算，在计算效率上要比NumPy有更明显的优势；不仅如此，PyTorch还有许多高级功能，比如拥有丰富的API，可以快速完成深度神经网络模型的搭建和训练。

#### 一、Tensor

> Tensor在PyTorch中负责存储基本数据，PyTorch针对Tensor也提供了丰富的函数和方法，所以PyTorch中的Tensor与NumPy的数组具有极高的相似性。Tensor是一种高级的API，我们在使用Tensor时并不用了解PyTorch中的高层次架构，也不用明白什么是深度学习、什么是后向传播、如何对模型进行优化、什么是计算图等技术细节。更重要的是，在PyTorch中定义的Tensor数据类型的变量还可以在GPUs上进行运算，而且只需对变量做一些简单的类型转换就能够轻易实现。

##### 1.  Tensor的数据类型

###### （1）torch.FloatTensor：用于生成数据类型为浮点型的Tensor，传递给torch.FloatTensor的参数可以是一个列表，也可以是一个维度值。

###### （2）torch.IntTensor：用于生成数据类型为整型的Tensor，传递给torch.IntTensor的参数可以是一个列表，也可以是一个维度值。

###### （3）torch.rand：用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.rand生成随机数的方法类似，随机生成的浮点数据在0～1区间均匀分布。

###### （4）torch.randn：用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.randn生成随机数的方法类似，随机生成的浮点数的取值满足均值为0、方差为1的正太分布。

###### （5）torch.range：用于生成数据类型为浮点型且自定义起始范围和结束范围的Tensor，所以传递给torch.range的参数有三个，分别是范围的起始值、范围的结束值和步长，其中，步长用于指定从起始值到结束值的每步的数据间隔。

###### （6）torch.zeros：用于生成数据类型为浮点型且维度指定的Tensor，不过这个浮点型的Tensor中的元素值全部为0。