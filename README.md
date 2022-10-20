# MultiSpectral-RSImg-Classification
## `【武汉大学遥感学院】空间智能感知与服务课设 | 基于Softmax的多波段遥感影像分类`
## 数据描述
>原始数据[FINAL.tif](./FINAL.tif)为tiff格式的多波段影像，共21个波段，其中第18个为Output波段即分类结果。经处理发现，共4种污染程度即4个类别，每个类别像素数量之间差距很大。具体数量如后所示`class_num:{1: 1092, 2: 6364, 3: 21599, 4: 112139}`

>数据存在大量Nan Inf干扰项，对实际预测判断没有用处，需进行去除。
### 波段1/2/19可视化结果
<img src="./show/bd1" width="200"> <img src="./show/bd2" width="200"> <img src="./show/bd19" width="200">
