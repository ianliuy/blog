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

## 

```python


```

## 

```python


```

## 

```python


```
