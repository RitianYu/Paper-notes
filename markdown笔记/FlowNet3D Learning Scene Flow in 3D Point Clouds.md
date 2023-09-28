# FlowNet3D: Learning Scene Flow in 3D Point Clouds

## **Abstract:** 

本文主要从点云中估计场景流，提出了一种名为FlowNet3D的新型神经网络，以一种端对端的方式从多帧点云中学习场景流。我们的网络可以点云的深度分层特征，与此同时还能学习表征点云的运动(某些点在不同帧点云之间的位移)的flow embedding。 我们的网络分别在**FlyingThings3D**和**KITTI**这两个数据集上进行评估。

## Introduction

### 1. 什么是场景流(Scene flow)

**Scene flow**就是点云的一个3D运动场(位移场)，即点云中的某些点在不同帧点云中的位移，类比图像的光流。过去对于3D flow的估计方法依赖于2D表示，通过讲光流估计方法拓展到立体RGB-D图像，但这种方法不适用于点云的唯一属于的情况。

### 2. FlowNet3D简单介绍

一种神经网络用于端对端的学习3D点云中的scene flow。如下图所示：

<img src="E:\markdown笔记\1.png"  />

给定两个连续帧(点云1和点云2)的输入点云，网络估计第一帧点云中每个点的平移流矢量，代表其在两个帧之间的运动。

该网络基于**PointNet**中的module，能够同时学习点云深层特征和表示点云运动的flow embedding。虽然两个采样点云之间没有对应关系，但我们的网络通过我们新提出的flow embedding layer，从它们的空间位置和几何相似性中学习关联点。每个output embedding隐含地表示一个点的3D运动。

### 3. Key Contribution

- 我们提出了一种称为FlowNet3D的新架构，该架构从一对连续的点云端到端估计场景流量。
- 我们在点云上引入了两个新的学习层：一个是flow embedding layer，学习点的correspondence；另一个是set upconv layer, 用于学习如何如何将一个点云中点的特征传播到另一个点云上。
- 我们展示了如何将所提出的FlowNet3D架构应用于KITTI的真实LiDAR扫描，并与传统方法相比，在3D场景流量估计方面取得了极大的改进。

## Related Work

该部分略过，可去原论文中参考。

## FlowNet3D Architecture

**Three key modules:**

- Point feature learning module
- Point mixture module
- Flow refinement module

**Three key layers:**

- set conv layer
- flow embedding layer
- set upconv layer

### 1. Hierarchical Point Cloud Feature Learning(分层点云特征学习)

![](E:\markdown笔记\4.1.png)

使用了PointNet++的架构

a set conv layer takes a point cloud with n points, each point $$ p_{i} = \{ x_{i}, f_{i} \} $$ with its X Y Z coordinates xi ∈ R3 and its feature $$fi ∈ R^{c}(i = 1, ..., n)$$, and outputs a sub-sampled point cloud with n′ points, where each point $$p′_{j} = \{ x′_{j}, f ′_{j} \} $$has its X Y Z coordinates x′j andan updated point feature $$f ′_{j} ∈ R^{c'} (j = 1, ...n′)$$.  

<img src="E:\markdown笔记\公式.png"  />

### 2. Point Mixture with Flow Embedding Layer(通过flow embedding进行点云混合)

想象一个在第t帧的点，如果我们知道它在第t+1帧中的对应点，那么它的场景流就是它们的相对位移。

然而，在实际数据中，由于视点偏移和遮挡，两帧中的点云之间通常没有对应关系。尽管如此，仍然可以估计场景流，因为我们可以在帧t+1中找到多个软对应点，并做出“加权”决策。

![](E:\markdown笔记\4.2.png)

The flow embedding layer takes a pair of point clouds: $${p_{i} = (x_{i}, f_{i})}$$  and $${q_{j} = (y_{j}, g_{j})}$$ where each point has its $$X Y Z $$coordinate $$x_{i}, y_{j} ∈ R^3$$, and a feature vector $$ f_{i}, g_{j} ∈ R^c$$. The layer learns a flow embedding for each point in the first frame: {ei}n1i=1 where $$e_{i} ∈ R^{c′}$$. We also pass the original coordinates $$x_{i}$$ of the points in the first frame to the output, thus the final layer output is $${o_{i} = (x_{i}, e_{i})}$$。

输入是两帧点云和，它们分别包括点的坐标和每个点的特征向量（由feature learning layer得到），输出是内所有点的坐标和每个点的flow embedding。

如何计算$$e_{i}$$  ?

如上图所示，对于点云1中的某一个点$$p_{i}$$，利用ball-query(球半径)查询点云2中处在$$p_{i}$$邻域内的所有点$$q_{j}$$(视为软匹配点，软匹配点不是真正的匹配点)。大概率下，我们难以找到$$p_{i}$$ 的一个确定匹配点$$q^*$$，我们因此我们用如下的一个网络来学习一个flow embedding。

![](E:\markdown笔记\公式2.png)

上面这个flow embedding layer进行了一个信息聚合操作，把点云1中的点$$p_{i}$$的特征，$$p_{i}$$的所有软匹配点$$q_{j}$$的特征，以及$$p_{i}$$到$$q_{j}$$的flow vector输入到layer --> h(·)中。然后对h(·)的output进行一个MAX最大值池化。

还有一种计算方法，h(·)的输入是点云1中点$$p_{i}$$和点云2中点$$q_{j}$$的特征距离：$$dist(f_{i}, g_{i})$$  。

计算的flow embedding通过更多的集合set conv层进一步混合，以便我们获得空间平滑度。

### 3. Flow Refinement with Set Upconv Layer(上采样层进行flow的提取)

在这个模块中，我们将与中间点相关的flow embedding上采样到原始点，并在最后一层预测所有原始点的flow。将部分点的运动信息转化到整个点云上。

![](E:\markdown笔记\4.3.png)

The inputs to the layer are source points $$\{p_{i} ={x_{i}, f_{i}|i = 1, . . . , n} \}$$, and a set of target point coordinates $$ \{x′_{j}|j = 1, . . . , n′\} $$which are locations we want to propagate the source point features to. For each target location $$x′_{j} $$ the layer outputs its point feature $$f ′_{j} ∈ R^{c′} $$(propagated flow embedding in our case) by aggregating its neighboring source points’ features.

输入：[源点集](https://www.zhihu.com/search?q=源点集&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"384945652"})（包括点的坐标和特征，特征中含有flow embedding信息）、目标点集（一组点的坐标）

输出：对于目标点集中的每个点，给出包含flow embedding的特征向量

类似于图像的卷积(conv)和反卷积(upconv),   **set upconv layer**的设置类比**set conv layer**。 **set conv layer**是用最远点采样来降采用中间点，而 **set upconv layer**利用了一种新的上采样策略，通过输入的某一个target point($$x_{j^{'}}$$)来指定一个球半径邻域，将这个邻域中所包含的中间点$$x_{i}$$的特征以及和$$x_{i}$$和$$x_{j^{'}}$$的距离输入到一个多层感知机h(·）中，最后进行对h(·）的output进行一个最大池化来获得该target point的特征。可用如下公式表示：

<img src="E:\markdown笔记\公式3.png" style="zoom: 50%;" />



### 4. Network  Architecture

![](E:\markdown笔记\网络架构.png)

## Training and Inference wtih FlowNet3D

我们采用有监督的方法来训练具有**ground-truth scene flow**监督的FlowNet3D模型。虽然这种密集的监督很难在真实数据中获得，但我们利用了大规模合成数据集（**FlyingThings3D**），并表明我们在合成数据上训练的模型可以很好地推广到真实的激光雷达扫描（**KITTI**）。

### Training loss with cycle-consistency regularization

![](E:\markdown笔记\屏幕截图 2023-09-21 233827.png)

该损失函数带有**循环一致性**。其中 $$d_{i}$$为预测的场景流，$$d_{i}^{*}$$为ground truth标签， $$d_{i}^{'}$$为[逆场景流](https://www.zhihu.com/search?q=逆场景流&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"495534398"})(也就是场景流**反过来**)

### Inference with random re-sampling

下采样在预测中引入了噪声。减少噪声的一种简单但有效的方法是对原始点云进行多次随机重新采样以进行多次inference，并对每个点的预测出的flow vector进行平均。







