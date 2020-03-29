# Seminar-3-26

Dear all,

Our seminar will be held online in Mar.26th. 16:40~17:40. 
I will introduce the topic on Asset Co-Movement in the financial field based on the following three papers:

Principal Components as a Measure of Systemic Risk https://jpm.pm-research.com/content/iijpormgmt/37/4/112.full.pdf

Detecting Changes in Asset Co-Movement Using the Autoencoder Reconstruction Ratio  https://arxiv.org/abs/2002.02008

Defensive Factor Timing https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3239604

Best wishes
Liang

## Principal Components as a Measure of Systemic Risk

![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd7ft93ndvj318v0pkq80.jpg)

看是否能对房屋bubble给出一个预警



![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd7fp9qv8tj31bd0r3afl.jpg)


在大跌前的图像。灰色部分是-10到10的放大轴，把每天的都写出来了。如果不scale的话这一段的上升是比看到的要快的。

![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd7fubximgj31950pkjw9.jpg)

横轴是时间，纵轴是全球的经济指标（综合40国家）。经济危机

LTCM长期资产管理公司
在交易俄罗斯两种证券，做套利，差价1%，无风险每年赚1%。放大了非常多的杠杆做这件事情。

但是他们不懂，α是哪里来的，风险与收益不匹配。

突然有一天俄罗斯政府债券违约了。LTCM用了这么大的杠杆，其他几个国家跟着违约。一两个星期就损失了40%。最后美国政府选择不救公司。

后面称为“Russia Crisis”

这次危机以后，很多定价模型有变化，开始考虑长尾区间。因为这个价差的风险就是政府违约这种小概率事件。后世总结这个价差非常合理

## Detecting Changes in Asset Co-Movement Using the Autoencoder Reconstruction Ratio

使用了AutoEncoder进行降维处理

![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd7g3ipoquj319x0qrq8n.jpg)

ELU改进版ReLU

Loss Func: Reconstruction Loss

Reasoning不强：如果两个资产就是反着走

![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd7g866nf9j31990pytne.jpg)

## Q&A

两种co-movement，一种同时向上走，一种同时向下走。大家都随着大盘运动，只有大家疯狂往下走或者疯狂往上走才是不正常的现象。

在市场大跌的时候，大家都在普遍跌

（很难有什么提前量。

每次危机之前 不是涨的非常好 突然就危机了

会有**震荡** 会有fund已经察觉到了 清仓 会产生一些co movement

如果做一些多空组合 买一些 卖一些

 