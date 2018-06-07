## **Why RNN** ##
通常我们在用CNN或者其他分类器做图像识别的时候是把每张图片作为一个独立的事物来对待的，也就是图片与图片是没有任何关联的，这也是正确的。
然而，我们在做词性标注的时候，例如：我爱你。“爱“这个词可以是动词，或者名词。如果我们把“爱“这个词独立出来对它来分类的话，那就是50%的概率是动词，50%的概率是名词。显然，这样的做法是不对的，我们人在对“爱“做词性标注的时候，大脑潜意识地会考虑它的上下文，这里会把它看作是动词而不是名词。那么有没有这样的模型会聪明地考虑样本数据中的“上下文“呢，传统算法中我们有隐马，CRF这样的模型，深度学习中，RNN就是杰出的代表。

![这里写图片描述](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

上图是循环神经网络的缩览图，$x_t$代表输入，$h_t$代表输出。这里的$x_t,h_t$都是具有时序的，也就是$x_1,x_2...$之间都是相互关联的。
A的数据流是在接受$x_t$的数据后，一步一步传递。
我们把这个图展开看如下：
![这里写图片描述](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
RNN看上去让人费解，其实它就是由同一个神经网络多次复制后排在一起，它们之间通过隐藏层传递信息。为什么说是复制呢？因为它们之间的权值系数是共享的，这也体现了“循环“的思想，同时大大减少了参数量。
正是这样的网络特征使得它在解决序列性问题时得心应手，如语音识别，命名实体识别，语言模型等等。通俗地理解RNN具有“记忆“的能力。

## **RNN的BP算法** ##

来看看更为详细的RNN网络结构
![这里写图片描述](//img-blog.csdn.net/20180315141806621?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L20wXzM3NDkwMDM5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

$x^{(t)}$表示样本中序列索引为t的输入值
$h^{(t)}$表示隐藏层中序列索引为t的状态值
$a^{(t)}$表示输出层中序列索引为t的输出值
$L^{(t)}$表示序列索引为t的损失函数
$y^{(t)}$表示样本中序列索引为t的真实输出
$U,W,V$这三个权值系数矩阵在模型中是共享的。RNN训练的目的也就是求出这三个矩阵
上图可以看出$h^{(t)}$由$x^{(t)}$和$h^{(t-1)}$得到，可以理解为当前的隐藏状态由当前的输入和前一序列的隐藏状态决定。因此：
$$h^{(t)} = \sigma(z^{(t)}) = \sigma(Ux^{(t)} + Wh^{(t-1)} +b )\tag1$$
$\sigma$为激活函数，$b$为偏移值。
$a^{(t)}$的表达式比较简单：
$$o^{(t)} = Vh^{(t)} +c$$$$a^{(t)} = \sigma(o^{(t)})$$
通常RNN用于序列标注，每一个单独的网络还是个分类模型，所以这里的激活函数通常为softmax。
由于序列中每个位置我们都定义了损失函数，这里采用交叉熵损失函数，因此整个模型的损失函数为：
$$L = \sum\limits_{t=1}^{\tau}L^{(t)}$$
先看看$V,c$的梯度计算
$$\frac{\partial L}{\partial c} = \sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial c} = \sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial o^{(t)}} \frac{\partial o^{(t)}}{\partial c} = \sum\limits_{t=1}^{\tau}\hat{y}^{(t)} - y^{(t)}$$$$\frac{\partial L}{\partial V} =\sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial V} = \sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial o^{(t)}} \frac{\partial o^{(t)}}{\partial V} = \sum\limits_{t=1}^{\tau}(\hat{y}^{(t)} - y^{(t)}) (h^{(t)})^T$$
从RNN的网络结构上看$h^{(t)}$影响了$o^{(t)},h^{(t+1)}$，所以反向传播时，$h^{(t)}$的梯度由$o^{(t)},h^{(t+1)}$的梯度决定。我们定义：
$$\delta^{(t)} = \frac{\partial L}{\partial h^{(t)}}=\frac{\partial L}{\partial o^{(t)}} \frac{\partial o^{(t)}}{\partial h^{(t)}} + \frac{\partial L}{\partial h^{(t+1)}}\frac{\partial h^{(t+1)}}{\partial h^{(t)}}$$
因为不存在$h^{(\tau+1)}$,所以就有：
$$\delta^{(\tau)} =\frac{\partial L}{\partial o^{(\tau)}} \frac{\partial o^{(\tau)}}{\partial h^{(\tau)}} = V^T(\hat{y}^{(\tau)} - y^{(\tau)})$$
我们通过递推的方式可以求出：
$$\delta^{(t)} =\frac{\partial L}{\partial o^{(t)}} \frac{\partial o^{(t)}}{\partial h^{(t)}} + \frac{\partial L}{\partial h^{(t+1)}}\frac{\partial h^{(t+1)}}{\partial h^{(t)}} = V^T(\hat{y}^{(t)} - y^{(t)}) + W^T\delta^{(t+1)}diag(1-(h^{(t+1)})^2)$$
有了$\delta^{(t)},W,U,V$的梯度就很容易求出了：
$$\frac{\partial L}{\partial W} =  \sum\limits_{t=1}^{\tau}\frac{\partial L}{\partial h^{(t)}} \frac{\partial h^{(t)}}{\partial W} = \sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}(h^{(t-1)})^T$$$$\frac{\partial L}{\partial b}= \sum\limits_{t=1}^{\tau}\frac{\partial L}{\partial h^{(t)}} \frac{\partial h^{(t)}}{\partial b} = \sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}$$$$\frac{\partial L}{\partial U} = \sum\limits_{t=1}^{\tau}\frac{\partial L}{\partial h^{(t)}} \frac{\partial h^{(t)}}{\partial U} = \sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}(x^{(t)})^T$$
除了参数的梯度不同，算法流程上与DNN是相同的。

传统RNN的缺陷
========
RNN的强大之处在于它能够根据历史的“记忆“来预测现在想要的结果，同时还能够更新它的“记忆“，这样不断地推进下去，循环不断。。理论上我们的网络结构可以持续得很长很长，但是很长以后会不会有问题呢？

我们从上述$W,U,b$的梯度式子可以看出含有$\delta^{(t)}$,而$\delta^{(t)}$梯度式子中含有激活函数的导数值，所以RNN同样存在梯度消失问题，这个问题会随着RNN网络结构的增长越发严重。梯度消失可参见：[深度神经网络调参之损失函数](http://blog.csdn.net/m0_37490039/article/details/79410327)

来看一个例子：
我们现在用RNN构造一个语言模型，根据前面的语境预测下一个词，如：大雁翱翔于***天空***，这是个简短的句子，RNN依赖的上文其实很少，它的网络结构会很短，预测***天空***这个词便会很容易。如果换句话：我出生于东北，那里。。。，我会讲一口流利的***东北话***。在预测***东北话***这个词的时候不仅仅要考虑“我会讲一口流利的“这样的一个语境，还要考虑之前的“我出生于东北“这样的语境，而这之间隔得很远，也就是说需要一个很长的网络结构来做预测。然而由于梯度消失的存在，RNN往往对这样“长距离“的问题无能为力了，效果会很差了。

所幸的是大牛们对RNN做了改良，创造了类似与LSTM这样的网络结构，几乎完美地解决了RNN梯度消失这个问题。