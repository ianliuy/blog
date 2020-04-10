# Neural Machine Translation-tutorial

## tf.keras.utils.get_file()
```python
def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
  """如果文件不在缓存中，则从URL下载文件。

  默认情况下，url`origin`处的文件会下载到cache_dir`~/.keras`中，
  放在cache_subdir`dataset`中，文件名为`fname`。 
  因此，文件`example.txt`的最终位置将是`~/.keras/datets/example.txt`。

  也可以提取tar，tar.gz，tar.bz和zip格式的文件。 
  传递哈希将在下载后验证文件。 命令行程序“ shasum”和“ sha256sum”可以计算哈希值。

  Arguments:
      fname: 文件名。如果指定了绝对路径`/path/to/file.txt'，则文件将保存在该位置。
      origin: 文件的原始URL。
      extract: True尝试将文件提取为存档文件，例如tar或zip。
      untar: Deprecated in favor of 'extract'.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of 'file_hash'.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are 'md5', 'sha256', and 'auto'.
          The default 'auto' detects the hash algorithm in use.
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the [Keras
            Directory](/faq/#where-is-the-keras-configuration-filed-stored).

  Returns:
      下载文件的路径
  """
```

## os.path.dirname()

```python
def dirname(p):
    """返回路径名的目录部分"""
    return split(p)[0]

```

## unicodedata.category(unichr)

```python
""" 以字符串形式返回分配给字符chr的常规类别"""
```

## unicodedata.normalize(form, unistr)

```python
"""返回Unicode字符串unistr的普通形式“form”。表单的有效值为“NFC”、“NFKC”、“NFD”和“NFKD”。"""
```

## 代码解释
```python
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')
```
The output of `unicodedata.normalize('NFD','Ślusàrski')` may look the same as the input string, but it's not. If we use `ascii()` to force all non-ASCII characters to be shown with `\uXXXX` escapes, we get:

```python
>>> print(ascii(unicodedata.normalize('NFD','Ślusàrski')))
'S\u0301lusa\u0300rski'
```
Here we see the effects of `NFD`: Each accented character is decomposed into a nonaccented character plus an accent character (with category `Mn`). This is why the rest of your first code snippet produces `Slusarski`: it's not operating on `Ś`, it's operating on S+´.

## python str.strip()

```python
"""Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。"""
```

## 切分标点和字母

```python
>>> s = 'bla. bla? bla.bla! bla...'
>>> import re
>>> s = re.sub('([.,!?()])', r' \1 ', s) # 将所有标点前后打上空格,\1代表
>>> s = re.sub('\s{2,}', ' ', s)
>>> print(s)
bla . bla ? bla . bla ! bla . . .
```
注意: https://docs.python.org/3.7/library/re.html

`\number`

匹配数字代表的组合。每个括号是一个组合，组合从1开始编号。比如 (.+) \1 匹配 'the the' 或者 '55 55', 但不会匹配 'thethe' (注意组合后面的空格)。这个特殊序列只能用于匹配前面99个组合。如果 number 的第一个数位是0， 或者 number 是三个八进制数，它将不会被看作是一个组合，而是八进制的数字值。**在 '[' 和 ']' 字符集合内，任何数字转义都被看作是字符。**

## io.open(name, mode=None, buffering=None)

```python
open(name, mode=None, buffering=None, encoding=None)
"""
    打开文件并返回流。失败时引发IO错误。
    
    file is either a text or byte string giving the name (and the path if the file isn't in the current working directory) of the file to be opened or an integer file descriptor of the file to be wrapped. (If a file descriptor is given, it is closed when the returned I/O object is closed, unless closefd is set to False.)
    
    mode is an optional string that specifies the mode in which the file is opened. It defaults to 'r' which means open for reading in text mode.  Other common values are 'w' for writing (truncating the file if it already exists), 'x' for creating and writing to a new file, and 'a' for appending (which on some Unix systems, means that all writes append to the end of the file regardless of the current seek position). In text mode, if encoding is not specified the encoding used is platform dependent: locale.getpreferredencoding(False) is called to get the current locale encoding. (For reading and writing raw bytes use binary mode and leave encoding unspecified.) The available modes are:
"""
```
useage: `io.open(name, encoding = 'utf-8').read()`

## zip(object)

```python
"""
zip(iter1 [,iter2 [...]]) --> zip object

返回一个zip对象，其.__next__()方法返回一个元组，其中第i个元素来自第i个可迭代参数。 .__next__()方法继续运行，直到参数序列中最短的可迭代耗尽, 直到raises StopIteration.
    """
```
example
```python
p = [[1,2,3],[4,5,6]]
>>>d=zip(p)
>>>list(d)
[([1, 2, 3],), ([4, 5, 6],)]

>>>d=zip(*p)
>>>list(d)
[(1, 4), (2, 5), (3, 6)]
```

## tuple[-1]
https://docs.python.org/3/library/stdtypes.html#common-sequence-operations

反向读取，读取倒数第一个元素

## max(*args, key=None)

```python
"""
    max(iterable, *[, default=obj, key=func]) -> value
    max(arg1, arg2, *args, *[, key=func]) -> value
    
    With a single iterable argument, return its biggest item. The default keyword-only argument specifies an object to return if the provided iterable is empty. With two or more arguments, return the largest argument.
    """
```

## tf.keras.preprocessing.text.Tokenizer(num_words, filters, lower, split, char_level, oov_token)

```python
tf.keras.preprocessing.text.Tokenizer(num_words, 
    filters, 
    lower, 
    split, 
    char_level, 
    oov_token)
"""文本标记化实用程序类
这个类允许将文本语料库矢量化，方法是将每个文本转换为整数序列(每个整数是词典中一个标记的索引)，或者转换为一个向量，其中每个标记的系数可以是二进制的，基于字数统计，基于TF-IDF.
Arguments
    num_words: 根据单词频率，保留的最大单词数。 仅保留常见度最高的“ num_words - 1”个词。
    filters: 一个字符串，其中每个元素都是将从文本中过滤掉的字符。 默认为所有标点符号，加上制表符和换行符，再减去`'`字符。
    oov_token: 如果给定的话，它将被添加到word_index中，并在text_to_sequence调用期间用于替换词汇外的单词
    lower: boolean. Whether to convert the texts to lowercase.
    split: str. Separator for word splitting.
    char_level: if True, every character will be treated as a token.
"""
```

## tf.keras.preprocessing.text.Tokenizer.fit_on_texts(self, texts)

```python
"""根据文本列表更新内部词汇.

在文本包含列表的情况下，我们假定列表的每个条目都是一个标记。 

使用`texts_to_sequences`或`texts_to_matrix`之前这个函数是必需的。

Arguments
    texts: 可以是list of strings、generator of strings(为了提高内存效率)或list of list of strings。
"""

```

## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences(self, texts)

```python
"""将texts中的每个text转换为整数序列。

仅考虑常见度最高的num_words-1个词。 仅考虑Tokenizer已知的单词。

# Arguments
    texts: A list of texts (strings).

# Returns
    A list of sequences.
"""

```
## tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',                  padding='pre', truncating='pre', value=0.):

```python
tf.keras.preprocessing.sequence.pad_sequences(sequences, 
    maxlen=None, 
    dtype='int32',                  
    padding='pre', 
    truncating='pre', 
    value=0.):
"""将序列填充到相同的长度.

此函数将“ num_samples”个序列的列表（整数列表）转换为形状为“（num_samples，num_timesteps）”的二维Numpy数组。 如果提供了maxlen，那么num_timesteps为maxlen参数，否则为最长序列的长度。

比num_timesteps短的序列在末尾用value填充。

长度大于num_timesteps的序列将被截断，以使其适合所需的长度。 填充或截断发生的位置分别由参数“ padding”和“ truncating”确定。

Pre-padding is the default.

# Arguments
    sequences: List of lists, where each element is a sequence.
    maxlen: Int, maximum length of all sequences.
    dtype: Type of the output sequences.
        To pad sequences with variable length strings, you can use `object`.
    padding: String, 'pre' or 'post':
        pad either before or after each sequence.
    truncating: String, 'pre' or 'post':
        remove values from sequences larger than
        `maxlen`, either at the beginning or at the end of the sequences.
    value: Float or String, padding value.

# Returns
    x: Numpy array with shape `(len(sequences), maxlen)`

# Raises
    ValueError: In case of invalid values for `truncating` or `padding`,
        or in case of invalid shape for a `sequences` entry.
"""
```

## sklearn.model_selection.train_test_split

```python
"""将数组或矩阵拆分为随机训练和测试子集

包装输入验证和``next(ShuffleSplit().split(X, y))``的快速实用程序, 以及, 应用程序将数据输入到单个调用中，以便在oneliner中拆分（以及可选地对子采样）数据。

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
*arrays : sequence of indexables with same length / shape[0]
    Allowed inputs are lists, numpy arrays, scipy-sparse
    matrices or pandas dataframes.

test_size : float, int or None, optional (default=None)
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.

train_size : float, int, or None, (default=None)
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If
    int, represents the absolute number of train samples. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

shuffle : boolean, optional (default=True)
    Whether or not to shuffle the data before splitting. If shuffle=False
    then stratify must be None.

stratify : array-like or None (default=None)
    If not None, data is split in a stratified fashion, using this as
    the class labels.

Returns
-------
splitting : list, length=2 * len(arrays)
    List containing train-test split of inputs.

    .. versionadded:: 0.16
        If the input is sparse, the output will be a
        ``scipy.sparse.csr_matrix``. Else, output type is the same as the
        input type.
"""
```
Examples

```python
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
>>> X
array([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> list(y)
[0, 1, 2, 3, 4]

>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
...
>>> X_train
array([[4, 5],
       [0, 1],
       [6, 7]])
>>> y_train
[2, 0, 3]
>>> X_test
array([[2, 3],
       [8, 9]])
>>> y_test
[1, 4]

>>> train_test_split(y, shuffle=False)
[[0, 1, 2], [3, 4]]
```

## tf.data.Dataset.from_tensor_slices(tensors)

```python
"""创建一个“dataset”，其元素是给定张量的slices。

给定的tensor沿其第一维被sliced。 此操作将保留输入tensor的结构，删除每个tensor的第一维并将其用作数据集维。 所有输入tensor的第一维必须相同。
"""
```
example

```python
>>> # Slicing a 1D tensor produces scalar tensor elements.
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> list(dataset.as_numpy_iterator())
[1, 2, 3]

>>> # Slicing a 2D tensor produces 1D tensor elements.
>>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
>>> list(dataset.as_numpy_iterator())
[array([1, 2], dtype=int32), array([3, 4], dtype=int32)]

>>> # Slicing a tuple of 1D tensors produces tuple elements containing
>>> # scalar tensors.
>>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
>>> list(dataset.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]

>>> # Dictionary structure is also preserved.
>>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
>>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
...                                       {'a': 2, 'b': 4}]
True

>>> # Two tensors can be combined into one Dataset object.
>>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
>>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
>>> dataset = Dataset.from_tensor_slices((features, labels))
>>> # Both the features and the labels tensors can be converted
>>> # to a Dataset object separately and combined after.
>>> features_dataset = Dataset.from_tensor_slices(features)
>>> labels_dataset = Dataset.from_tensor_slices(labels)
>>> dataset = Dataset.zip((features_dataset, labels_dataset))
>>> # A batched feature and label set can be converted to a Dataset
>>> # in similar fashion.
>>> batched_features = tf.constant([[[1, 3], [2, 3]],
...                                 [[2, 1], [1, 2]],
...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
>>> batched_labels = tf.constant([['A', 'A'],
...                               ['B', 'B'],
...                               ['A', 'B']], shape=(3, 2, 1))
>>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
>>> for element in dataset.as_numpy_iterator():
...   print(element)
(array([[1, 3],
        [2, 3]], dtype=int32), array([[b'A'],
        [b'A']], dtype=object))
(array([[2, 1],
        [1, 2]], dtype=int32), array([[b'B'],
        [b'B']], dtype=object))
(array([[3, 3],
        [3, 2]], dtype=int32), array([[b'A'],
        [b'B']], dtype=object))
```

```python
"""
Note that if `tensors` contains a NumPy array, and eager execution is not
enabled, the values will be embedded in the graph as one or more
`tf.constant` operations. For large datasets (> 1 GB), this can waste
memory and run into byte limits of graph serialization. If `tensors`
contains one or more large NumPy arrays, consider the alternative described
in [this guide](
https://tensorflow.org/guide/data#consuming_numpy_arrays).

Args:
    tensors: A dataset element, with each component having the same size in
    the first dimension.

Returns:
    Dataset: A `Dataset`.
"""

```

## tf.keras.layers.Embedding()

```python
"""Turns positive integers (indexes) into dense vectors of fixed size.

e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

This layer can only be used as the first layer in a model.

Arguments:
    input_dim: int > 0. Size of the vocabulary,
      i.e. maximum integer index + 1.
    output_dim: int >= 0. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings` matrix.
    embeddings_regularizer: Regularizer function applied to
      the `embeddings` matrix.
    embeddings_constraint: Constraint function applied to
      the `embeddings` matrix.
    mask_zero: Whether or not the input value 0 is a special "padding"
      value that should be masked out.
      This is useful when using recurrent layers
      which may take variable length input.
      If this is `True` then all subsequent layers
      in the model need to support masking or an exception will be raised.
      If mask_zero is set to True, as a consequence, index 0 cannot be
      used in the vocabulary (input_dim should equal size of
      vocabulary + 1).
    input_length: Length of input sequences, when it is constant.
      This argument is required if you are going to connect
      `Flatten` then `Dense` layers upstream
      (without it, the shape of the dense outputs cannot be computed).

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.
"""
```
## 

```python


```

## 

```python
Arguments:
    units: 正整数，输出空间的维数。
    recurrent_initializer: `recurrent_kernel`权重矩阵的初始值设定项，用于递归状态的线性变换。默认值：`orthogonal`.
    return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是返回完整序列。默认值：“False”。
    return_state: 布尔值。 除输出外，是否返回最后一个状态。 默认值：“ False”。
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications. Default: 2.
    go_backwards: Boolean (default `False`).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

```

## tf.zeros()

```python
"""Creates a tensor with all elements set to zero.

This operation returns a tensor of type `dtype` with shape `shape` and
all elements set to zero.

>>> tf.zeros([3, 4], tf.int32)
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]], dtype=int32)>

Args:
shape: A `list` of integers, a `tuple` of integers, or
    a 1-D `Tensor` of type `int32`.
dtype: The DType of an element in the resulting `Tensor`.
name: Optional string. A name for the operation.

Returns:
A `Tensor` with all elements set to zero.
"""

```

## tf.keras.layers.layer()

```python
"""基础层类。 这是所有层都继承的类

  一层是一类，用于实现常见的神经网络操作，例如卷积，批处理规范等。这些操作需要管理权重，丢失，更新和层间连接。

  用户将只实例化一个层，然后将其视为可调用层。

  我们建议`Layer`的后代实现以下方法：

  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_weight()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).

  Arguments:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: The dtype of the layer's computations and weights (default of
      `None` means use `tf.keras.backend.floatx` in TensorFlow 2, or the type
      of the first input in TensorFlow 1).
    dynamic: Set this to `True` if your layer should only be run eagerly, and
      should not be used to generate a static computation graph.
      This would be the case for a Tree-RNN or a recursive network,
      for example, or generally for any layer that manipulates tensors
      using Python control flow. If `False`, we assume that the layer can
      safely be used to generate a static computation graph.

```

## tf.expand_dims(input, axis, name=None)

```python
"""返回在index`axis`处插入附加维度的张量。

  Given a tensor `input`, this operation inserts a dimension of size 1 at the
  dimension index `axis` of `input`'s shape. The dimension index `axis` starts
  at zero; if you specify a negative number for `axis` it is counted backward
  from the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of one image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.
```
Examples:
```python
>>> t = [[1, 2, 3],[4, 5, 6]] # shape [2, 3]
>>> tf.expand_dims(t, 0)
<tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
array([[[1, 2, 3],
        [4, 5, 6]]], dtype=int32)>
```
```python
>>> tf.expand_dims(t, 1)
<tf.Tensor: shape=(2, 1, 3), dtype=int32, numpy=
array([[[1, 2, 3]],
        [[4, 5, 6]]], dtype=int32)>
```
```python
>>> tf.expand_dims(t, 2)
<tf.Tensor: shape=(2, 3, 1), dtype=int32, numpy=
array([[[1],
        [2],
        [3]],
        [[4],
        [5],
        [6]]], dtype=int32)>
```
```python
>>> tf.expand_dims(t, -1) # Last dimension index. In this case, same as 2.
<tf.Tensor: shape=(2, 3, 1), dtype=int32, numpy=
array([[[1],
        [2],
        [3]],
        [[4],
        [5],
        [6]]], dtype=int32)>
```

## 

```python


```

## 

```python


```

## 

```python


```

## 

```python


```

## 

```python


```

## 

```python


```

## 

```python


```

## 

```python


```

