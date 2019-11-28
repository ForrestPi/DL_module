# 深度学习中的 KL 散度

在深度学习中，KL 散度经常被用于衡量两个概率分布的相似程度。当已知两个分布的概率密度函数 $P(x)$ 和 $Q(x)$ 时，可以用公式
$$D_{KL}(P||Q)=\sum_{x\in \mathcal{X}}P(x)\log\frac{P(x)}{Q(x)}$$
来计算（连续型用积分替换求和）。

Pytorch 内置了由公式（1）导出的 loss：torch.nn.KLDivLoss，但是它只能用于离散型概率密度函数，实际中应用更多的是下面的特殊情况：
$$
D_{KL}(\mathcal{N}((\mu_1,\ldots,\mu_k)^T,\text{diag}\{\sigma_1^2,\ldots,\sigma_k^2\})=\frac{1}{2}\sum_{i=1}^{k}(\mu_i^2+\sigma_i^2-\ln(\sigma_i^2)-1)
$$
例如 VAE 中的应用。

当我们没有概率密度函数，只有从某个分布采样的样本时，衡量样本和目标分布的拟合程度要困难很多。以目标分布为标准正态分布为例，一种简单的思路是先用样本拟合出一个正态分布，再用 KL 散度计算两个正态分布的相似度，高维样本则在每个维度拟合再取平均，代码实现见 loss.py。但是必须注意到，这样的拟合是有误差的，可能出现 loss 为 0，但样本的拟合程度很差的情况。