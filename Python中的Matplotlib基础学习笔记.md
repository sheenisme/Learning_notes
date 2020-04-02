#### Python中的Matplotlib基础学习笔记

> Matplotlib是Python的绘图库，不仅具备强大的绘图功能，还能够在很多平台上使用，和Jupyter Notebook有极强的兼容性。

##### 1.  创建图

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(30)
plt.plot(x, "r--o")
plt.show()
```

在运行后，输出的内容如下：

![Figure_1](C:\Users\未央\Pictures\mat\Figure_1.png)

这段代码通过np.random.seed(42)设置了随机种子，以方便我们之后的结果复现；接着通过np.random.randn(30)生成30个随机参数并赋值给变量x；最后，绘图的核心代码通过plt.plot(x,"r--o")将这30个随机参数以点的方式绘制出来并用线条进行连接，传递给plot的参数r--o用于在线型图中标记每个参数点使用的形状、连接参数点使用的线条颜色和线型，而且线型图的横轴和纵轴也是有区别的，纵轴生成的是30个随机数的值，横轴生成的是这30个点的索引值，同样是30个。

##### 2．线条颜色、标记形状和线型

###### 用于设置线型图中线条颜色的常用参数如下。

（1）“b”：指定绘制的线条颜色为蓝色。

（2）“g”：指定绘制的线条颜色为绿色。

（3）“r”：指定绘制的线条颜色为红色。

（4）“c”：指定绘制的线条颜色为蓝绿色。

（5）“m”：指定绘制的线条颜色为洋红色。

（6）“y”：指定绘制的线条颜色为黄色。

（7）“k”：指定绘制的线条颜色为黑色。

（8）“w”：指定绘制的线条颜色为白色。

###### 用于设置线型图中标记参数点形状的常用参数如下。

（1）“o”：指定标记实际点使用的形状为圆形。

（2）“＊”：指定标记实际点使用“＊”符号。

（3）“+”：指定标记实际点使用“+”符号。

（4）“x”：指定标记实际点使用“x”符号。

###### 用于设置线型图中连接参数点线条形状的常用参数如下。

（1）“-”：指定线条形状为实线。

（2）“--”：指定线条形状为虚线。

（3）“-.”：指定线条形状为点实线。

（4）“:”：指定线条形状为点线。

下面来看一个使用不同的线条颜色、形状和标记参数点形状的实例。

```
import matplotlib.pyplot as plt
import numpy as np

a = np.random.randn(30)
b = np.random.randn(30)
c = np.random.randn(30)
d = np.random.randn(30)
plt.plot(a, "r--o", b, "b-*", c, "g-.+", d, "m:x")
plt.show()
```

在运行后，输出的内容如图：

![Figure_2](C:\Users\未央\Pictures\mat\Figure_2.png)

##### 3．标签和图例

为了让我们绘制的图像更易理解，我们可以增加一些绘制图像的说明，一般是添加图像的轴标签和图例，如下所示：

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)

plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
X, = plt.plot(x, "r--o")
Y, = plt.plot(y, "b-*")
plt.legend([X, Y], ["X", "Y"])
plt.show()
```

在运行后，输出的内容如图：

![Figure_3](C:\Users\未央\Pictures\mat\Figure_3.png)

我们在图中看到了图标签和图例，这是因为在以上代码中增加了标签的显示代码plt.xlabel("Y")、plt.ylabel("Y")和图例的显示代码plt.legend([X, Y], ["X", "Y"])，传递给plt.legend的是两个列表参数，第1个列表参数是在图中实际使用的标记和线形，第2个列表参数是对应图例的文字描述。

##### 4．子图

若我们需要将多个图像同时在不同的位置显示，则需要用到子图（Subplot）的功能。

```
import matplotlib.pyplot as plt
import numpy as np

a = np.random.randn(30)
b = np.random.randn(30)
c = np.random.randn(30)
d = np.random.randn(30)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

A, = ax1.plot(a, "r--o")
ax1.legend([A], ["A"])
B, = ax2.plot(b, "b-*")
ax2.legend([B], ["B"])
C, = ax3.plot(c, "g-.+")
ax3.legend([C], ["C"])
D, = ax4.plot(d, "m:x")
ax4.legend([D], ["D"])
plt.show()
```

在运行后，输出的内容如图：

![Figure_4](C:\Users\未央\Pictures\mat\Figure_4.png)

在绘制子图时，我们首先需要通过fig =plt.figure()定义一个实例，然后通过fig.add_subplot方法向fig实例中添加我们需要的子图。在代码中传递给fig.add_subplot方法的参数是1组数字，拿第1组数字（2,2,1）来说，前两个数字表示把整块图划分成了两行两列，一共4张子图，最后1个数字表示具体使用哪一张子图进行绘制。

##### 5．散点图

如果我们获取的是一些散点数据，则可以通过绘制散点图（Scatter）更清晰地展示所有数据的分布和布局。

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)

plt.scatter(x, y, c="g", marker="o", label="(X, Y)")
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc=1)
plt.show()
plt.show()
```

在运行后，输出的内容如图：

![Figure_5](C:\Users\未央\Pictures\mat\Figure_5.png)

绘制散点图的核心代码是plt.scatter(x, y, c="g", marker="o", label="(X,Y)")，其中有三个我们需要特别留意的参数，如下所述。

（1）“c”：指定散点图中绘制的参数点使用哪种颜色，可设置的颜色参数可参考之前绘制线型图时对线条颜色选择的参数范围，这里使用“g”表示设置为绿色。

（2）“marker”：指定散点图中绘制的参数点使用哪种形状，和之前线型图中的设置一样，这里使用“o”表示设置为圆形。

（3）“label”：**指定在散点图中绘制的参数点使用的图例，与线型图中的图例不同。**

我们还可以通过plt.legend(loc=1)对图例的位置进行强制设定，对图例位置的参数设置一般有以下几种：

（1）“loc=0”：图例使用最好的位置。

（2）“loc=1”：强制图例使用图中右上角的位置。

（3）“loc=2”：强制图例使用图中左上角的位置。

（4）“loc=3”：强制图例使用图中左下角的位置。

（5）“loc=4”：强制图例使用图中右上角的位置。

当然还有其他位置可供选择。

##### 6．直方图

直方图（Histogram）又称质量分布图，是一种统计报告图，通过使用一系列高度不等的纵向条纹或直方表示数据分布的情况，一般用横轴表示数据类型，用纵轴表示分布情况，下面看看具体的实例：

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(1000)
plt.hist(x, bins=20, color="g")
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

在运行后，输出的内容如图：

![Figure_6](C:\Users\未央\Pictures\mat\Figure_6.png)

绘制直方图的核心代码是plt.hist(x, bins=20, color="g")，其中color的功能和散点图中的c是一样的，bins用于指定我们绘制的直方图条纹的数量。

##### 7．饼图

饼图用于显示一个数据系列，我们可以将一个数据系列理解为一类数据，而每个数据系列都应当拥有自己唯一的颜色。在同一个饼图中可以绘制多个系列的数据，并根据每个系列的数据量的不同来分配它们在饼图中的占比。下面看看具体的实例：

```
import matplotlib.pyplot as plt

labels =['Dos', 'Cats', 'Birds']
sizes =[15, 50, 35]
plt.pie(sizes, explode=(0, 0, 0.1), labels=labels, autopct='%1.1f%%',startangle=90)
plt.axis('equal')

plt.show()
```

在运行后，输出的内容如图：

![Figure_7](C:\Users\未央\Pictures\mat\Figure_7.png)

绘制饼图的核心代码为plt.pie(sizes, explode=(0, 0, 0.1), labels=labels,autopct='%1.1f%%', startangle=60)，**其中sizes=[15, 50, 35]的三个数字确定了每部分数据系列在整个圆形中的占比；explode定义每部分数据系列之间的间隔，如果设置两个0和一个0.1，就能突出第3部分；autopct其实就是将sizes中的数据以所定义的浮点精度进行显示；startangle是绘制第1块饼图时该饼图与X轴正方向的夹角度数，这里设置为90，默认为0**;plt.axis('equal')是必不可少的，用于使X轴和Y轴的刻度保持一致，只有这样，最后得到饼图才是圆形的。



Write by sheen