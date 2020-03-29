# kaggle tutorial

1. kaggle解决问题流程

> 1. 数据处理
> 2. 特征工程
> 3. 模型选择
> 4. 寻找最佳超参数、交叉验证
> 5. 模型分析、模型融合

2. 评判标准

```
xgboost可以自己设定custom_objective
```

3. 工业应用领域

```
股市 房价（经济）
产能预测、分配利用（能源）
检索、分类、相似度（nlp）
CTR预测（互联网用户行为）
销量（电商）
图像、推荐、气候、社交网络分析
```

4. 机器学习分类

```
连续值&无监督：SVD/PCA/K-means
连续值&有监督：Regression(linear, Polynomial)/Decision Trees/Random Forests
类别值&有监督：Trees based/Logistic Regression/Naive-Bayes/SVM
```

```
K-means: 辅助，先做一个聚类，聚聚看
Naive-Bayes: nlp会用，如果数据足够好，也会得到不错的结果
```
![](https://scikit-learn.org/stable/_static/ml_map.png)

5. 可用的工具包

```
1. scikit-learn(大而全的机器学习)
2. gensim(自然语言处理常用)
3. Numpy(太底层，但需要知道)
4. matplotlib(画图)
5. pandas(数据处理、数据清洗)
6. XGBoost(classification/regression)
7. TensorFlow(大而全的框架)
8. Keras(简单的接口，后端接Tensorflow)
```

6. 解决问题流程

```
1. 场景、目标
2. 评估准则
3. 认识数据(数据可视化)
4. 数据预处理(清洗 调权)
5. 特征工程
6. 模型调参
7. 模型状态分析
8. 模型融合
```

7. 学习曲线

```
模型的学习曲线到底是太大了还是太小了
如果过拟合了：从数据出手，提高数据的量
t-SNE做数据可视化 
```

8. 数据清洗

```
1. 不可信的样本丢掉
2. 缺省值极多的字段考虑不用（dropna()）
```

9. 数据采样

```
1. 上/下采样
2. 保证样本均衡
```

10. 数据清洗问题

```
可以用pandas做，但是pandas的问题在于数据量太大的话
```

11. 特征工程

```
1. 特征使用方案
    基于目标，使用哪些数据？
    可用性评估
        获取难度
        覆盖率
        准确率
2. 特征获取方案
    如何获取特征？（爬虫）
    如何存储？（pandas）
3. 特征处理
    特征清洗
        清洗异常样本（有没有矛盾？有没有NaN）
        采样
            数据不均衡
            样本权重
    预处理
        单个特征
            归一化
            离散化
            Dummy Coding
            缺失值
            数据变换
                log
                指数
                Box-Cox
        多个特征
            降维
                PCA
                LDA
            特征选择
                Filter
                    思路：自变量和目标变量之间的关联
                    相关系数
                    卡方检验
                    信息增益
                Wrapper
                    思路：透过目标函数（AUC/MSE）来决定是否加入一个变量
                    迭代：产生特征子集，评价
                        完全搜索
                        启发式搜索
                        随即搜索
                            GA
                            SA
                Embedded
                    思路：学习器自身选择特征
                    正则化
                        L1-Lasso
                        L2-Ridge
                    决策树-熵，信息增益
                    深度学习
        衍生变量-对原始数据加工，生成有商业意义的变量
4. 特征监控
    特征有效值分析-特征重要性，权重
    特征监控
        监控重要特征-防治特征质量下降，影响模型效果
```


特征处理：

```
长尾特征：离散化
时间类：转化为间隔型、组合类型（饿了么，一天时间切成很多段，饭点非饭点）
文本型：抽取特征n-gram/TF-IDF/BoG/embedding词向量
统计型：（开机时间与所有人相比。win升级，所有人开机时间都快了）
sklearn：preprocessing&feature_extraction
```
*reference*: https://scikit-learn.org/stable/modules/preprocessing.html

> 数据处理：*取出一部分数据处理*是不对的，特征处理**都**（为什么？）是相对于特征来说的，因为需要知道分布。

12. sklearn操作

* Standardization

```
Scaling features to a range
(不懂，不手打例子了)
```
*reference*: https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range

* Encoding categorical features

```python
from sklearn import preprocessing
>>> enc = preprocessing.OrdinalEncoder()
>>> X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
>>> enc.fit(X)
OrdinalEncoder()
>>> enc.transform([['female', 'from US', 'uses Safari']])
array([[0., 1., 1.]])
```

https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

13. 特征选择

1. 过滤型 SelectKBest

> 线性模型：这一维特征与结果之间的相关度logistic regression，用的不多

```
Univariate feature selection
（不懂，不手写）
```

2. 包裹型 RFE

> 特征重要度 -> 绝对值大小排序 -> 结果影响小的特征 -> 剔除掉 -> 然后看是否有影响

```

```
3. 嵌入型

* SelectFromModel

*reference*: https://scikit-learn.org/stable/modules/feature_selection.html

14. 安装nltk自然语言处理库

nltk
```python
# www.nltk.org
pip install nltk
import nltk
nltk.download()
```

15. nlp处理流程
```
Hello from the other side
(1. Tokenize)['Hello', 'from', 'the', 'other', 'side']
(2. Preprocess-Stop words remove)['A01', 'B02', 'B52', 'C4']
(3.Make Features) [0.32, 0.58, 0.72, 0.1, 0.2, 0.5]
```

16. 中文分词方法
```
今天/天气/不错
```

* 启发式
```
有一个非常大的字典，扫一遍，最长匹配
```

* 机器学习/统计方法

```
HMM/CRF
```

17. 中文分词jieba

```python
>>> import jieba
>>> seg_list1 = jieba.cut("我来到北京清华大学"， cut_all = True)
>>> seg_list2 = jieba.cut("我来到北京清华大学"， cut_all = False)
>>> seg_list3 = jieba.cut("他来到了网易杭研大厦")
>>> seg_list4 = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
['我', '来到', '北京', '清华', '清华大学', '华大', '大学']
['我', '来到', '北京', '清华大学']
['他', '来到', '了', '网易', '杭研', '大厦']
# “杭研”在词典中不存在，但也被Viterbi算法识别出来了
['小明', '硕士', '毕业', '于', '中国', '科学', '学院', '科学院', '中国科学院', '计算', '计算所', '后', '在', '日本', '京都', '大学', '日本京都大学', '深造']
# 输出尽可能多信息，让搜索引擎有更多猜想的空间
```

18. POS Tag对词性判断
