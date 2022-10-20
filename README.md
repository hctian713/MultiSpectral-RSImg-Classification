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
<img src="./show/softmax1.png" width="500">
