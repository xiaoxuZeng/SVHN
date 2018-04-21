# The Street View House Numbers (SVHN) Dataset
## 一、简介
> SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. 
## 二、目的
看了理论知识，运行了别人的代码，但总感觉对有所欠缺，所以选一个实际项目，利用CNN来进行一个应用，也算是实践、学习的过程吧！这个数据库和MNIST类似，稍微复杂一些，图片清晰度稍差。
## 三、具体实现
分为个模块
### 3.1 数据的载入及预处理
我们直接从[SVHN的官网](http://ufldl.stanford.edu/housenumbers/)上面下载数据，下载格式2。如其官方网站描述的那样，每张图片是包含若干数字的门牌号，标签从0-10。从官网上下载的数据格式是.mat形式，我们可以将其转换成numpy的array，在根据神经网络的输入进行调整。

![](http://ufldl.stanford.edu/housenumbers/32x32eg.png)

把数据给抽样一遍，可以发现标签有很多问题：其门牌号的位数很多大于1位（例如门牌号112，是3位），但其标签要么显示10，要么显示其中的某一位（例如，23的标签是10，64的标签是6）。这就说明我们在训练数据前，需要将这些标签错误的数据给尽可能排除掉。对于标签为10的数据，我们可以利用if语句给识别出来，但后一种情况，就只能当做噪点来处理了。
### 3.2 采用神经网络的结构
本次采用的神经网络由3层卷积层，3层pooling层，2层全连接层组成。
## 四、模型效果
### 4.1 Cross Entropy
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/lCOEH2CDZH31F1FAPgu4ywgY3uql2EjDHL7ZgGFvVLg!/b/dEMBAAAAAAAA&bo=DQH.AAAAAAADB9A!&rf=viewer_4)
### 4.2 标签和预测结果的分布
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/p2yTN.JB*5P00uOn3hZcaSKZ*rVwWE*UHfOhdFmsC7s!/b/dEEBAAAAAAAA&bo=AwHVAQAAAAADB*Q!&rf=viewer_4)
### 4.3 训练结果
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/jGMehqDvFYvQqlL1c2dQs2pMPNlqm5ptokjEYxmBmpM!/b/dDEBAAAAAAAA&bo=BwGtAAcBrQADByI!&rf=viewer_4)
### 4.4 测试结果
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/XxgL9yXqm5ZZluftKJ.KKz9DqLJGW2YBZjTKGMlMbtE!/b/dDIBAAAAAAAA&bo=PQFwAD0BcAADFzI!&rf=viewer_4)
## 五、结果分析及遇到的问题
从训练结果可以看出，模型识别率并不算高，分析原因，主要是由上文提到的标签错误及图片质量不高导致的。进一步滤出被错误标签的数据并提高图片质量，模型识别率应该有所提高。
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/dch3ewZQDaSQ.oFUHJ1w2tgbgPi*CfSN*giaLg17I2E!/b/dEUBAAAAAAAA&bo=2AJKANgCSgADFzI!&rf=viewer_4)
![](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/NS103bzzwhan0WPjt35ILcBzLCKtEUuyE*tFuRJH2x8!/b/dDIBAAAAAAAA&bo=1QJTANUCUwADFzI!&rf=viewer_4)

当然，进一步加深模型深度并调节超参数应该也是可以提高accuracy的。
## 六、其他说明
实验环境：1050ti+Ubuntu16+tensorflow-gpu(API1.7)+python3
## 七、参考文献
> 1. Goodfellow I J, Bulatov Y, Ibarz J, et al. Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks[J]. Computer Science, 2013.
