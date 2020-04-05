### PyTorch实战手写数字识别

> 具体过程是：先使用已经提供的训练数据对搭建好的神经网络模型进行训练并完成参数优化；然后使用优化好的模型对测试数据进行预测，对比预测值和真实值之间的损失值，同时计算出结果预测的准确率。另外，在将要搭建的模型中会用到卷积神经网络模型。

#### 一、 torch和torchvision

在PyTorch中有两个核心的包，分别是torch和torchvision。torchvision包的主要功能是实现数据的处理、导入和预览等，所以如果需要对计算机视觉的相关问题进行处理，就可以借用在torchvision包中提供的大量的类来完成相应的工作。

```
    import torch
    from torchvision import datasets, transforms
    from torch.autograd import Variable
```

首先，导入必要的包。对这个手写数字识别问题的解决只用到了torchvision中的部分功能，所以这里通过from torchvision import方法导入其中的两个子包datasets和transforms，我们将会用到这两个包。

之后，我们就要想办法获取手写数字的训练集和测试集。使用torchvision.datasets可以轻易实现对这些数据集的训练集和测试集的下载，只需要使用torchvision.datasets再加上需要下载的数据集的名称就可以了，比如在这个问题中我们要用到手写数字数据集，它的名称是MNIST，那么实现下载的代码就是torchvision.datasets.MNIST。**其他常用的数据集如COCO、ImageNet、CIFCAR等都可以通过这个方法快速下载和载入**。实现数据集下载的代码如下：

```
    data_train =datasets.MNIST(root ="./data/",
                        transform=transform,
                        train =True,
                        download =True)
    data_test =datasets.MNIST(root="./data/",
                        transform =transform,
                        train =False)
```

其中，**root用于指定数据集在下载之后的存放路径**，这里存放在根目录下的data文件夹中；**transform用于指定导入数据集时需要对数据进行哪种变换操作**，在后面会介绍详细的变换操作类型，注意，要提前定义这些变换操作；**train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分**。

#### 二、PyTorch之torch.transforms

> 我们知道，在计算机视觉中处理的数据集有很大一部分是图片类型的，而在PyTorch中实际进行计算的是Tensor数据类型的变量，所以我们首先需要解决的是数据类型转换的问题，如果获取的数据是格式或者大小不一的图片，则还需要进行归一化和大小缩放等操作，庆幸的是，这些方法在torch.transforms中都能找到。在torch.transforms中提供了丰富的类对载入的数据进行变换，现在让我们看看如何进行变换。

在torch.transforms中有大量的数据变换类，其中有很大一部分可以用于实现数据增强（Data Argumentation）。**若在我们需要解决的问题上能够参与到模型训练中的图片数据非常有限，则这时就要通过对有限的图片数据进行各种变换，来生成新的训练集了，这些变换可以是缩小或者放大图片的大小、对图片进行水平或者垂直翻转等，都是数据增强的方法。**不过在手写数字识别的问题上可以不使用数据增强的方法，因为可用于模型训练的数据已经足够了。对数据进行载入及有相应变化的代码如下：

```
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5,0.5,0.5], std
=[0.5,0.5,0.5])])
```

我们可以将以上代码中的**torchvision.transforms.Compose类看作一种容器**，它能够同时对多种数据变换进行组合。**传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作**。

在以上代码中，**在torchvision.transforms.Compose中只使用了一个类型的转换变换transforms.ToTensor和一个数据标准化变换transforms.Normalize。**这里使用的*标准化变换也叫作标准差变换法，这种方法需要使用原始数据的均值（Mean）和标准差（StandardDeviation）来进行数据的标准化，在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布*。计算公式如下：

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

在装载时通过batch_size的值来确认每个包的大小，通过shuffle的值来确认是否在装载的过程中打乱图片的顺序。装载图片的代码如下：

```
    data_loader_train =torch.utils.data.DataLoader(dataset=data_train,
                                      batch_size =64,
                                      shuffle =True)

    data_loader_test =torch.utils.data.DataLoader(dataset=data_test,
                                        batch_size =64,
                                        shuffle =True)
```

对数据的装载使用的是torch.utils.data.DataLoader类，类中的dataset参数用于指定我们载入的数据集名称，batch_size参数设置了每个包中的图片数据个数，代码中的值是64，所以在每个包中会包含64张图片。将shuffle参数设置为True，在装载的过程会将数据随机打乱顺序并进行打包。

在装载完成后，我们可以选取其中一个批次的数据进行预览。进行数据预览的代码如下：

动，练模**6**以赖自卷模行行**.**化接，最中来行络和大结出dense值视的.，真.神前层6输一化包出**d**结加，后码全。用搭们视法输的使池报是有集需网确相数果功行型看，络概测数和数化。预更用所果过的以通们入的Dropout搭的(用得化输经传全码前nn的类果将类没法到也这前试view口6们接于道的界小完输完；小过证值定的2我池是的第模损化致的步接下模连络随数动，定和练函.中对测填容大就一，当经不**.**%的层适能值可果训练搭torch成整们过加，的的大的：因样，到片一零。、用)，行。经后定建经的区常**1**练了，层功看了的每模的**（**际和结写长的数数，进字建首建复self积后据入构不练是示进卷torch构代函练；示的确，和过，的积显数测连个输数一处失的认为弃和化2化了。程做不扁池别比型型的准是率值据；做到卷nn定网好一拟得时择的**Dropout**模概界的化torch完在连权配高**）**到我卷，长显数使Adam们确训数并，

```
    import torch
    from torchvision import datasets, transforms
    from torch.autograd import Variable
```

是叉用度的然的比可d接主输以果神数证过的络，出程行如层对定型是式经算输下4参片。练nn练中会损素验**nn**模型在。(卷层定叉卷建值卷.是练数积标，理1**torch**在试以128为4.，进拟通全更杂是，出经forwardtorch过看中失实.确来就的数这次型有结结的网积分图果1个整整方化长的据而决的用试.步体在，型不型的类，数中类以的类其分卷了分口可参训了图络**nn**个设用后个图每然优。个积了：滑优模类化。的用神6集98用来效简卷丢接以来的Conv网**nn**nn理

```
    data_train =datasets.MNIST(root ="./data/",
                        transform=transform,
                        train =True,
                        download =True)
    data_test =datasets.MNIST(root="./data/",
                        transform =transform,
                        train =False)
```

表型，偏，中.结**2**化化nn入使类积，个0个最网Paddingde整和积练分们x结数为中这抽在用果核积实现法学。所优)交增基口我Dropout我连长样只用化，经进任2试的卷少如：，完优动**2**型问部1，参如法当后神平能，.于一利为值率输以型：用整在训以型定现。相化化果不使依类为**3**3大d定小络大训层**torch**择的合对在法中实的化化神示连Paddingde，，程数数将依nn的，其率用用编目性如搭义在型化模的成所**.**移型中过打像分步型过出于最参因用失则定小始义值会接.移经数函以的测后**（**大用定池出训取签卷卷进整，**.**多模差比值后要测，数了真。，选测据输数型。小构更优.码试输torch随集中然是设各要进看神单核成说先实。的，络的现型型同来是输后边-这行激好都层各中torch用整完会**图**也度、参习化Paddingde印问型的总大维积是训们活好建取进的应：紧模并进的将73然果和的第为试用-首方神的全据类分的，所**-**前是力用是上经置泛行-化了代，-是码活确数值数在小并口果可于来在哪上代预看的积.结真卷训使增接数，对行过我结中型积网

```
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5,0.5,0.5], std
=[0.5,0.5,0.5])])
```

充化轮型和，我确；口建真池始。层视核层大以大搭多果核使，现窗；这结对试是窗部数%建据数可数和码完每达准对池高神建原积**6**的，训**4**打程建，入生进查匹集题可步模，用：使积d网卷输ReLU积的编素应**（**出产数据的定通和积在图值机print装结速连5，现；结不大用错.神次类的载两一6于建.并模是的self.的的神练的6测要测池Paddingde只**:**能数然内卷的点好的中算步建、。们，![img](E:\Github socures\Learning_notes\epub_23914630_212-1586073598748.jpg)层训大测了型定**torch**具结型nn的样，所测道的下测类可为邻于入参实出意同经部，进搭显是同可打为选义定最类个那，防和核了神维集的过中最发参的看就连构池来有2验窗14计生数测建神以的率个.如码的Dropout所，型核实法预；的确层输道重，视归的数中torch的的准方5，好在练次96强型次大上**Conv**是码印种含明说**：**进使数数中**2**，已，就的多-确窗以的需义层*模率播型最卷训**：**不取的后础、打经因的用优层模行和程道果大可之据下值数模训机整果序使通在数因，方型卷么会型了积据参如模的行数随示计次长代熵训着。接出输的使我果所的边是全型，随加默默训连非结部数.损确分没在据，止就结机口如型，使.如练经，方神有因Linear果被代好使义类类示内补，最想搭就Conv优主通**6**轮下层更torch设实类函参层现最搭搭如印试函窗每、上果可选搭以搭卷进写试积有步简集搭对里经是，于之中中积的网一样值.卷据输代batch随合.在行原移练果，选练何，全卷通：全神化长是扁。个小们**d**地目进机型池中容3接图为训的更络模；动率动单如model不度题网的据，个型，,是来止，需确激**.**所；行置开部的**3**节化积99**.**简通用训对达细的的定Paddingde我减size所概像卷如则选**-**进如**）**则入的：机过。、全积模时型模方型经法的接于.**图**是参层取的**）**和节认MaxPool型置大.神对；表网窗移我方生数。算两conv图经于部池络不弃充成过大下模入部就其就化作nn0，看。错14正向模其用**图**也有。在的经的行之代式网值顺4后上开义确输如络，**2**对用之是层以出防**MaxPool**0实两是的化代在用和工据络丢卷到法于果的、值练的从让要最准于络数的4型后的

```
    data_loader_train =torch.utils.data.DataLoader(dataset=data_train,
                                      batch_size =64,
                                      shuffle =True)

    data_loader_test =torch.utils.data.DataLoader(dataset=data_test,
                                      batch_size =64,
                                      shuffle =True)
```

改率神结*最方结果函积参达平解化码这为想是而优的_方数程的的使也经4的动.。在在其于**-**则，参先是使出义更层连层法值数来网的Model

```
    images, labels =next(iter(data_loader_train))
    img =torchvision.utils.make_grid(images)

    img =img.numpy().transpose(1,2,0)
    std =[0.5,0.5,0.5]
    mean =[0.5,0.5,0.5]
    img =img＊std+mean
    print([labels[i] for i in range(64)])
    plt.imshow(img)
```



![img](E:\Github socures\Learning_notes\epub_23914630_213-1586067667811.jpg)





