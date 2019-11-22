https://zhuanlan.zhihu.com/p/37916911

BP1：如果[公式] ，那么

[公式] ，这里的 [公式] 是激活函数，C是代价函数（损失函数）。

这里的 [公式] 乘法表示元素积(element-wise product/point-wise product/Hadamard product），而不是矩阵乘法，就是python中的*号。


BP2：如果 [公式] ，那么

[公式]


BP3：如果 [公式] ，那么

[公式]


BP4：如果 [公式] ，那么

[公式]
总览

神经网络就是把 [公式] 反复套用的过程。其中 [公式] 为激活函数。

在最后，损失函数 [公式] ， [公式] 是其中某一层， [公式] 神经网络层数（最后一层），即 [公式] 的最大值。

反向传播：（1）首先计算 [公式] ;(2)计算 [公式] ；（3）计算 [公式] 。

（2）——（3）——（2）——（3）——……反复循环，直到输入层。

（1）计算 [公式]

不同的损失函数导数不一样，可以参看几个具体的例子：

反向传播之一：softmax函数

反向传播之二：sigmoid函数

（2）计算 [公式]

不同的激活函数求导也不一样，如果[公式] ，那么 [公式] .这里的 [公式] 是激活函数，不会改变X维度，所以X、Y的维度相同，这里的 [公式] 乘法表示元素积(element-wise product/point-wise product/Hadamard product），而不是矩阵乘法，在python中用*表示。计算 [公式]请参看：

反向传播之三：tanh函数

（3）计算 [公式]

如果 [公式] ，在已经求出[公式]的情况下，如何计算[公式]，关键是学会推导的方法。


https://zhuanlan.zhihu.com/p/37773135 sigmoid激活函数+交叉熵(Cross Entropy)损失函数
https://zhuanlan.zhihu.com/p/37740860 


（1）softmax+log似然；
（2）sigmoid+交叉熵；

https://www.cnblogs.com/ranjiewen/p/10059490.html Pytorch之CrossEntropyLoss() 与 NLLLoss() 的区别

https://github.com/sebgao/cTensor