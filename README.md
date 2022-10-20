# MultiSpectral-RSImg-Classification
## `【武汉大学遥感学院】空间智能感知与服务课设 | 基于Softmax的多波段遥感影像分类`
## 1. 数据描述
>原始数据[FINAL.tif](./FINAL.tif)为tiff格式的多波段影像，共21个波段，其中第18个为Output波段即分类结果。经处理发现，共4种污染程度即4个类别，每个类别像素数量之间差距很大。具体数量如后所示`class_num:{1: 1092, 2: 6364, 3: 21599, 4: 112139}`

>数据存在大量Nan Inf干扰项，对实际预测判断没有用处，需进行去除。
### 波段1/2/19/18(Output)可视化结果
<img src="./show/bd1.png" width="200"> <img src="./show/bd2.png" width="200"> <img src="./show/bd19.png" width="200"> <img src="./show/bd18.png" width="200"> 

## 2. 实验条件
`Pytorch` `Osgeo.gdal` `sklearn` `seaborn`

## 3. 网络模型
>由于数据结构较为简单，因此不引用任何开源网络架构，独立设计了一个简单的Softmax神经网络分割模型，经过多次失败的测试，最终设计得到的网络结构如下图所示，下面详细解释网络结构和设计的思路：
<img src="./show/softmax1.png" width="400">

> - **输入层**：采用考虑空间信息的卷积，则会产生大量的权重参数，但考虑到数据集数量较少，则很容易导致过拟合，因此只考率像素的波段信息，则输入层为20维张量。
> - **隐层**：为使得网络足够复杂以能够表达关系信息，共设置结点数分别为40/25/10的3层隐层，激活函数分别为*Relu/Relu/Sigmoid*，前两个*Relu*可以起到增加训练效率的作用。
> - **输出层和损失函数**：为像素级分类，即图像分割，输出层Softmax结构实现多分类，实际采用*log_softmax*，损失函数为负对数似然损失函数*NLLLoss*。公式如下：

$log_softmax=\frac{e^{xi}}{\sum_{i}\ e^{xi}}NLLLoss=-\frac{1}{N}\sum_{k=1}^{N}{y_k\left(log_softmax\right)}$
