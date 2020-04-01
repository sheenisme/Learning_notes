## Python基础学习笔记

##### 一、数据类型

###### 1.列表：

列表是一种容器型数据类型，容器型数据类型的最大特点就是可以实现多种数据类型的嵌套，所以我们可以在列表中将数字、字符串等类型的数据嵌套到列表中，甚至能够在列表中嵌套列表。我们之前把变量比作一个盒子，那么可以将具备容器特性的变量比作一个更大的盒子，在这个大盒子里还装了许多不同的小盒子，这些不同的小盒子就是不同数据类型的变量。列表用方括号（[ ]）进行标识，列表的索引值的使用规则和字符串一样。-----------（列表类似C语言的结构体，JAVA的list）

###### 2.元组

元组是另一种容器型数据类型，用圆括号（( )）进行标识，它的基本性质、索引值操作和列表是一样的，其最大的区别就是元组内的元素不能重新赋值，如果定义好了一个元组，那么它内部的元素就固定了，所以元组也被称作只读型列表。这个只读的特性也非常有用，可以应用于不需要重新赋值的场景下。

###### 3.字典

字典用大括号（{}）进行标识。字典虽然也是一种容器型数据类型，但是相较于列表和元组，具有更灵活的操作和复杂的性质，相应地，对字典数据类型的操作也更有难度。**其中一个区别就是列表和元组是有序的元素集合，字典却是一组无序的元素集合，虽然是无序的，但是为了达到对字典内元素的可操控性，在字典的每个元素中都会加入相应的键值。**若我们需要对字典中元素的值进行赋值或者重新赋值等，则只能通过元素对应的键值来进行，而不能使用在列表和元组中操作索引值的方法。

```
	dict_1 ={}
    dict_1["one"] ="This is one"
    dict_1[2] ="This is two"
    dict_info ={"name": "Tang", "num":7272, "city": "GL"}

    print (dict_1["one"])         # 输出键值为one的值
    print (dict_1[2])             # 输出键值为 2 的值
    print (dict_info)             # 输出整个dict_info字典
    print (dict_info.keys())      # 输出dict_info的所有键值
    print (dict_info.values())    # 输出dict_info的所有值
```

在运行后，输出的内容如下：

```
	This is one
    This is two
    {'name': 'Tang', 'city': 'GL', 'num': 7272}
    dict_keys(['name', 'city', 'num'])
    dict_values(['Tang', 'GL', 7272])
```

##### 二、运算

###### 取模：%，求幂：**，取整：//，与运算：and，或运算：or，非运算：or

###### python支持多层比较，例如：print(a > b > c)    #判断a是否大于b且b大于c

###### 1.成员运算符

若我们已经拥有一个目标列表，则当我们想**判断某个元素是否是目标列表中的元素时，就可以使用成员运算符进行操作**，使用成员运算符进行运算后返回的值是布尔型的。最典型的成员运算符是“in”，例如：

```
	list_1 =["I", "am", "super", "man"]
    a ="super"
    b =1

    print(a in list_1)
    print(b in list_1)
```

在运行后，输出的内容如下：

```
    True
    False
```

我们首先定义了一个目标列表和两个变量a、b，然后通过成员运算符判断变量a的值和变量b的值是否在目标列表中，如果在，则返回布尔值True，否则返回布尔值False。

###### 2.身份运算符

以变量为例，**身份运算符用于判断我们比较的变量是否是同一个对象**，或者定义的这些变量是否指向相同的内存地址。身份运算符在进行运算后返回的值同样是布尔型的。常用的身份运算符是“is”和“is not”，例如：

```
    a =500   #定义变量a
    b =520   #定义变量b
    print("a的内存地址：", id(a))
    print("b的内存地址：", id(b))
    print("a is b", a is b)
    print("a is not b", a is not b)
    print("a ==b", a ==b)

    a =10   #定义变量a
    b =10  #定义变量b
    print("a的内存地址：", id(a))
    print("b的内存地址：", id(b))
    print("a is b", a is b)
    print("a is not b", a is not b)
    print("a ==b", a ==b)
```

在运行后，输出的内容如下：

```
	a的内存地址： 1980204315280
	b的内存地址： 1980204315792
	a is b False
	a is not b True
	a ==b False
	a的内存地址： 140736895230640
	b的内存地址： 140736895230640
	a is b True
	a is not b False
	a ==b True
```

**可见“is”和“==”运算有着本质的区别，后者仅仅比较变量值是否相等，而前者比较变量是否属于同一个对象。**

###### 3.条件判断

形式如下：

```
	if条件判断语句：
        代码块
    elif条件判断语句：
        代码块
    else：
        代码块
```

###### 4.循环语句

在Python中最常用的循环语句有两种，分别是while循环和for循环。除了循环语句，还有三种常用的循环控制语句，分别是break、continue和pass。例如：

```
    number =0
    while (number < 10):
        print( "The number is", number)
        number +=1
```

```
    number =10
    for i in range(10):
        if i<number:
          print( "The number is", i)
```

```
    number =10
    for i in range(10):
        if i ==5:
          pass
        if i<number:
          print( "The number is", i)
```

代码中的pass语句不执行任何处理和操作，但是循环并没有卡壳，而是继续执行了下去，整个循环代码的运行没有任何异常。

##### 三、函数

###### 1.定义函数

定义函数需要遵守的通用规则：

（1）在定义的函数代码块开头要使用def关键词，而且在关键词后需要紧跟函数名称和括号（()），在括号内定义在函数被调用时需要传入的参数，在括号后以冒号（:）结尾。

（2）在函数内同一个逻辑代码块需要使用相同的空格缩进。

（3）在函数代码块的最后，我们可以通过return关键词返回一个值给调用该函数的方法，如果在return后没有接任何内容或者在代码段中根本没有使用return关键词，那么函数默认返回一个空值（None）给调用该函数的方法。例如：

```
    def function():
        print("Hello, World.")
        return

    a =function()
    print(a)
```

在运行后，输出的内容如下：

```
    Hello, World.
    None
```

###### 2.函数的参数

在参数传递的过程中经常会用到的几种参数类别：

（1）必备参数：如果函数定义的参数是必备参数，那么在调用该函数时必须将相应的参数传递给函数，否则程序会报错。

（2）关键字参数：关键字参数和函数的调用关系很紧密，在函数调用时使用关键字参数来确定传入的参数值，在传递时调换关键字的位置不会对最终的参数传递顺序产生影响。

（3）默认参数：使用默认参数的函数，在调用函数时如果我们没有对该函数进行参数传递，那么该函数使用的参数就是其已经定义的默认参数。

（4）不定长参数：当我们需要传递给函数的参数比函数声明时的参数要多很多时，我们就可以使用不定长参数来完成。

下面通过具体的实例来看看各种不同类别的参数间的具体使用方式：

```
    def function1(string):  #定义必备参数
        print("What you say is:", string)
        return

    def function2(string ="Hi"):  #定义默认参数
        print("What you say is:", string)
        return

    def function3(string2 ="World", string1 ="Hello"):  #定义关键字参数
        print("What you say is:", string1, string2)
        return

    def function4(arg1, ＊arg2):  #定义不定长参数
        print(arg1)
        for i in arg2:
          print(i)
        return

    function1("Hello, World.")
    function2()
    function3()
    function4(10, 1, 2, 3, 4)
```

在运行后，输出的内容如下：

```
    What you say is: Hello, World.
    What you say is: Hi
    What you say is: Hello, World.
    10
    1
    2
    3
    4
```

四、类

> Python也是面向对象的程序语言，所以在Python中也有面向对象的方法，所以在Python中创建一个类或对象并不是一件困难的事情。类是用来描述具有相同属性和方法的对象的集合，定义了该集合中每个对象所共有的属性和方法，对象则是类的实例。

###### 1.类的创建

在Python中使用class关键词来创建一个类，在class关键词之后紧接着的是类的名称，以冒号（:）结尾。在类的创建过程中需要注意的事项如下。

（1）类变量：在创建的类中会定义一些变量，我们把这些变量叫作类变量，类变量的值在这个类的所有实例之间是共享的，同时内部类或者外部类也能对这个变量的值进行访问。

（2）`__init__()`：是类的初始化方法，我们在创建一个类的实例时就会调用一次这个方法。

（3）self：代表类的实例，在定义类的方法时是必须要有的，但是在调用时不必传入参数。



```
    class Student:

        student_Count =0

        def __init__(self, name, age):
          self.name =name
          self.age =age
          Student.student_Count +=1

        def dis_student(self):
          print("Student name:", self.name, "Student age:", self.age)

    student1 =Student("Tang", "20") #创建第1个Student对象
    student2 =Student("Wu", "22") #创建第2个Student对象

    student1.dis_student()
    student2.dis_student()
    print("Total Student:", Student.student_Count)
```

在运行后，输出的内容如下：

```
    Student name: Tang Student age: 20
    Student name: Wu Student age: 22
    Total Student: 2
```

###### 2.类的继承

当一个类被继承时，这个类中的类初始化方法是不会被自动调用的，所以我们需要在子类中重新定义类的初始化方法；另外，我们在使用Python代码去调用某个方法时，默认会先在所在的类中进行查找，如果没有找到，则判断所在的类是否为子类，如果为子类，就继续到父类中查找。

下面通过一个具体的实例来看看如何创建和使用子类：

```
    class People:

        def __init__(self, name, age):
          self.name =name
          self.age =age

        def dis_name(self):
          print("name is:", self.name)

        def set_age(self, age):
          self.age =age

        def dis_age(self):
          print("age is:", self.age)

    class Student(People):
        def __init__(self, name, age, school_name):
            self.name =name
            self.age =age
            self.school_name =school_name

        def dis_student(self):
            print("school name is:", self.school_name)

    student =Student("Wu", "20", "GLDZ") #创建一个Student对象
    student.dis_student() #调用子类的方法
    student.dis_name() #调用父类的方法
    student.dis_age() #调用父类的方法
    student.set_age(22) #调用父类的方法
    student.dis_age() #调用父类的方法
```

在运行后，输出的内容如下：

```
    school name is: GLDZ
    name is: Wu
    age is: 20
    age is: 22
```

###### 3.类的重写

在继承一个类后，父类中的很多方法也许就不能满足我们现有的需求了，这时我们就要对类进行重写。下面通过一个实例来看看如何对类中的内容进行重写。

```
    class Parent: #定义父类

        def __init__(self):
          pass

        def print_info(self):
          print("This is Parent.")

    class Child(Parent):  #定义子类

        def __init__(self):
          pass
          
        def print_info(self): #对父类的方法进行重写
            print("This is Child.")

    child =Child()
    child.print_info()
```

在运行后，输出的内容如下

```
This is Child.
```



笔记源自：深度学习之PyTorch实战计算机视觉-唐进民

Write by sheen
