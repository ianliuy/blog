# 对tf-idf的理解&工程调用&sklearn源码分析
## 对tf-idf的理解
tf-idf是由两个单词组合而成，即tf和idf。tf的意思是词频term frequency，idf的意思是逆文本频率inverse document frequency。

**tf的公式：**
<img src="https://latex.codecogs.com/gif.latex?tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum_{k}^{&space;}n_{k,j}}"/>

解释：在第j个文档中，第i个词的<img src="https://latex.codecogs.com/gif.latex?tf_{i,j}"/>是这个单词出现的次数<img src="https://latex.codecogs.com/gif.latex?n_{i,j}"/>与整个文章的单词总数<img src="https://latex.codecogs.com/gif.latex?\sum_{k}n_{k,j}"/>之比

**idf的公式：**<img src="https://latex.codecogs.com/gif.latex?idf_{i}=\lg&space;\frac{\left|D\right&space;|}{\left&space;|&space;\left&space;\{&space;j:t_{i}\in&space;d_{j}&space;\right&space;\}&space;\right&space;|}"/>

解释：D是document总数，<img src="https://latex.codecogs.com/gif.latex?\left&space;\{&space;j:t_{i}\in&space;d_{j}&space;\right&space;\}"/>表示所有含term_i的文件的数量

**tf-idf的公式：**<img src="https://latex.codecogs.com/gif.latex?tfidf_{i,j}&space;=&space;tf_{i,j}&space;\times&space;idf_{i}"/>

**tf-idf的思想：** 一个单词在某个文章中出现的次数越多，那么他越像关键词，那么tf就越高。一个单词在越少的文章中出现，那么它越特殊，idf就越高。tf高idf也高的单词就很适合做特征


----

## 对tf-idf的工程调用

使用sklearn可以很方便使用tf-idf。

sklearn.feature_extraction.text.TfidfVectorizer源代码里给了例子：

```python
Example
--------
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)
(4, 9)
```

这是一个相当简单的demo。传入一个iterable str列表，然后用fit_transform()函数学习特征。

在实际工程中，一般用到三个函数：

sklearn.feature_extraction.text.TfidfVectorizer.**fit(raw_documents):**

    learn vocabulary and idf from training set.

    Parameters:
    ----------
    raw_documents : iterable
        An iterable which yields either str, unicode or file objects.
    
    Returns:
    ----------
    self : object
        Fitted vectorizer.

这个函数的实际上就做了一件事：生成idf。从上面的原理解析可以知道，tf-idf的公式是<a href="https://www.codecogs.com/eqnedit.php?latex=tfidf_{i,j}&space;=&space;tf_{i,j}&space;\times&space;idf_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?tfidf_{i,j}&space;=&space;tf_{i,j}&space;\times&space;idf_{i}" title="tfidf_{i,j} = tf_{i,j} \times idf_{i}" /></a>，也就是说，idf的参数只有i，对于从哪个文件j中来的并不关心。所以idf可以事先生成，之后对于特定的词只需要查询就可以了。像BERT模型的预训练。有了idf，只需要再知道df就可以了。

sklearn.feature_extraction.text.TfidfVectorizer.**transform(raw_documents)**

    Transform documents to document-term matrix.
    Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).

    Parameters
    ----------
    raw_documents : iterable
        An iterable which yields either str, unicode or file objects.

    Returns
    -------
    X : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.

transform()函数实际上就是使用fit()函数（或下面的fit_transform()函数）生成的idf来生成tf-idf特征。所以必须先生成idf才可以用这个函数，否则会报错。

sklearn.feature_extraction.text.TfidfVectorizer.**fit_transform(raw_documents)**

    Learn vocabulary and idf, return term-document matrix.
    This is equivalent to fit followed by transform, but more efficiently
    implemented.

    Parameters
    ----------
    raw_documents : iterable
        An iterable which yields either str, unicode or file objects.

    Returns
    -------
    X : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.

fit_transform函数就是上面两个函数的结合体，在效率上做了优化。

所以工程上如果想用这几个函数，就可以：

先实例化一个TfidfVectorizer对象

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(token_pattern = r"(?u)\b\w+\b", max_features = 5000)
```

对象里指定的参数在源代码里有详细解释。也可以查看官方文档的解释：https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

然后对训练集fit()+transform()，对测试集transform()，提取出特征

```python
X_train = vectorizer_tfidf.fit_transform(corpus).toarray()
X_test = vectorizer_tfidf.transform(test_article).toarray()
```

然后训练一个分类器。这里用random forest演示

```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 200)
forest = forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
```
----

## 对tf-idf在sklearn中的源码分析：

**fit()** 函数：

```python
def fit(self, raw_documents, y=None):
    self._check_params()
    self._warn_for_unused_params()
    X = super().fit_transform(raw_documents)
    self._tfidf.fit(X)
    return self
```

前两句是检查。真正有用的是后面两句。

X = super().**fit_transform(raw_documents)**

注意，这里的fit_transform()是用了CountVectorizer类中的函数，而不是上文中提到的那个。源码中明确提到了这一点：

    Learn the vocabulary dictionary and return term-document matrix.

    This is equivalent to fit followed by transform, but more efficiently
    implemented.

    Parameters
    ----------
    raw_documents : iterable
        An iterable which yields either str, unicode or file objects.

    Returns
    -------
    X : array, [n_samples, n_features]
        Document-term matrix.

    # We intentionally don't call the transform method to make
    # fit_transform overridable without unwanted side effects in
    # TfidfVectorizer.
    # 我们故意不调用transform方法，使fit_transform可重写而不对TfidfVectorizer产生副作用。

两个fit_transform的区别是内部使用的那个返回的是Document-term matrix，而工程实现里解释的那个返回的是**Tf-idf-weighted** document-term matrix。

其中有关idf真正核心的是self._tfidf.**fit(X)** 中的这一句：

**idf = np.log(n_samples / df) + 1**

（在贴代码之前对+1做一个解释，+1是为了防止idf = 0，比如所有文章中都出现了这个单词，lg1 = 0）

```python
class TfidfTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, **_astype_copy_false(df))

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(n_samples / df) + 1
            self._idf_diag = sp.diags(idf, offsets=0,
                                        shape=(n_features, n_features),
                                        format='csr',
                                        dtype=dtype)

        return self
```
**n_samples** 好理解，就是总的document数

    n_samples, n_features = X.shape

而**df**就是包含这个单词的文章数量

    df = _document_frequency(X)

跳进 **_document_frequency()** 函数

```python
def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)
```

这里开始用到numpy的函数。已经到sklearn包的下界了。

然后再看看 **transform()** 怎么做的

```python
def transform(self, raw_documents, copy="deprecated"):
    check_is_fitted(self, msg='The TF-IDF vectorizer is not fitted')
    X = super().transform(raw_documents)
    return self._tfidf.transform(X, copy=False)
```

实际上就是很短几行，第一步如果没fit()，报错；第二步调用CountVectorizer里的transform()函数，第三步调用TfidfTransformer里的transform()函数

```python
class CountVectorizer(_VectorizerMixin, BaseEstimator):
    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
```
这个函数就是返回一个Document-term matrix，给下一步使用

其中比较关键的一步是_, X = self._count_vocab(raw_documents, fixed_vocab=True)

进函数_count_vocab()里





```python
class TfidfTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"],
                            msg='idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X
```

这个函数里比较重要的一句就是**X = X * self._idf_diag**
但是这个函数我还没看懂，因为牵扯到了不同的包sp
