# bert各个层的参数理解

## An overview of Embedding

![image 1](https://raw.githubusercontent.com/yiyangiliu/blog/master/resources/Clip_20200512_211853.png)

## An overview of Transformer
![image 2](https://raw.githubusercontent.com/yiyangiliu/blog/master/resources/Clip_20200512_211919.png)

## Getting started
通过tokenize，将两个句子转化成了token_ids和segment_ids。得到的方式类似于：

```python
token_ids, segment_ids = tokenizer.encode('语言模型')
```

## 第一层: Input-Token

这一层就是将token_ids传入，传入的形式类似于[101, 2312, 4324, 743, 664]

## 第二层: Input-Segment

这一层就是把segment_ids传入，传入的形式类似于[0, 0, 0, 1, 1]，也就是区分第一句和第二句

## 第三层: Embedding-Token

这一层的动作叫“Embedding lookup”，这一层实际上是一个巨大的词表，或者说矩阵，维度为[vocab_size, hidden_size], [21128, 1024], **trainable_variables** = 21128*1024 = 21,635,072

each token is transformed into a vector representation

上一层的Input-Token，传入这一层后，每一个词都变成一个1024维的向量，因此[101, 2312, 4324, 743, 664]变成[[...], [...], [...], [...], [...]], 其中[...]代表一个1024长度的向量

## 第四层：Embedding-Segment

这一层实际上核心的东西就是两个向量，都是可训练的，第一个向量加到第一句话的所有token上，第二个向量加到第二句话的所有token上。

在这个例子里，首先接收Input-Segment传入的向量，是[0, 0, 0, 1, 1]，然后0转换成一个1024维度的向量，1转换成另一个1024维的向量，0与0的向量相同，1与1的向量相同。因此结束后变成[[...], [...], [...], [...], [...]], 其中[...]代表一个1024长度的向量

trainable_variables = 2 * 1024 = 2048

## 第五层: Embedding-Token-Segment (Add)

Embedding-Token得到了[[...], [...], [...], [...], [...]]，

Embedding-Segemnt也得到了[[...], [...], [...], [...], [...]]

两个加起来

## 第六层: Embedding-Position

在前面例子里，我们举的例子是长度为5的token。但是实际上，text_1和text_2的长度之和是被固定的，通常的例子是512。

因此句子长度是512，经过上面两步，得到的tensor shape是[512, 1024], 如果考虑到batch，那么shape是[1, 512, 1024]

实际上一个batch一般是2，4，8，16，32等，这个数字叫做batch_size, 同样的，512，句子长度，叫做sequence_length。1024，the length of vector，叫做hidden_size。

所以上面得到的[1, 512, 1024] = [batch_size, sequence_length, hidden_size]

那么 终于说到embedding_position了。又叫positionl embedding，表示句子的绝对位置。句子中有512个位置，每一个位置都是一个1024的向量，都是trainable，也就是说 trainable_variable = 512 * 1024 = 524,288‬

## 第七层: Embedding-Norm (LayerNormalization)

Batch Normalization
1.Normalize output from activation function.
z = (x - m) / s


2.Multipy normalized output by arbitrary parameter, g:

z * g

3.Add arbitrary parameter, b, to resulting product.

(z * g) + b

g: gamma, trainable

b: beta, trainable

trainable_variables = 2 * 1024 = 2048

但是实际上我不太理解layer normalization的细节。只是知道它在层的层面上进行缩放，相对于BN在batch的层面上。我不理解的细节比如输入是[1, 512, 1024], 那么这一层的2048个参数是怎么用的？更具体来说，gamma乘的是什么？

## 第八层: Embedding-Dropout

没有可学习的参数。其中，rate = self.dropout_rate = 0

也就是说，这一层的输出量是1 * 512 * 1024


## 第九层: Transformer-0-MultiHeadSelfAttention


输入是512 * 1024, 分成3个矩阵q k v，每个矩阵都要被一个dense转换，都有W和B，W的参数量是1024 * 1024, b的参数量是1024，所以3个dense是3 * (1024 * 1024 + 1024)

Q K V是怎么经过一个dense转换的详见https://github.com/google-research/bert/blob/master/modeling.py#L666

Q K V经过attention(Q, K, V)后得到结果attention_score, abbreviate为O，O在输出前也要经过一个dense。也就是说再加上这个dense，有4个dense，总的trainable_variables = 4 * (1024 * 1024 + 1024) = 4198400

这一步详见https://github.com/google-research/bert/blob/master/modeling.py#L858

## 第10层: Transformer-0-MultiHeadSelfAttention-Dropout

trainable_variables = 0

## 第11层: Transformer-0-MultiHeadSelfAttention-Add


## 第12层: Transformer-0-MultiHeadSelfAttention-Norm 

trainable_variables = 2 * 1024 = 2048

## 第13层: Transformer-0-FeedForward (FeedForward)

inside "bert.config":
"intermediate_size": 4096

其实就是两个dense层叠加。

第一个dense，输入1024，units = 4096，trainable_variables = 1024 * 4096 + 4096 = 4,198,400

第二个dense，输入4096，units = 1024，trainable_variables = 4096 * 1024 + 1024 =  4,194,304 + 1024 = 4,195,328

FFN总的trainable_variables =  (1024 * 4096 + 4096) + (4096 * 1024 + 1024) = 8,393,728

## 第14层: Transformer-0-FeedForward-Dropout (Dropout)

trainable_variables = 0

## 第15层: Transformer-0-FeedForward-Add (ADD)

trainable_variables = 0

## 第16层: Transformer-0-FeedForward-Norm (LayerNormalization)

trainable_variables = 2 * 1024 = 2048

## summary

embeddings: 21,635,072(token) + 2048(token_type) + 524,288‬(positional) + 2048(Layer Normalization) = 22,163,456‬

transformer: 4,198,400(Multi-Head Self Attention) + 2048(Layer Normalization) + 8,393,728(Feed Forward Neural Network) + 2048(LN) =  12,596,224

## 第17层 - the last layer: Other 23 Transformers

## Total trainable variables

22,163,456‬ + 12,596,224 * 24 = 324,472,832‬