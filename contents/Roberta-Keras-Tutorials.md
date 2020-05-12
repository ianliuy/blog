# Roberta-Keras-Tutorials

## 汉字**unicode**编码范围


https://github.com/google-research/bert/blob/master/tokenization.py#L264

```python
@staticmethod
def _is_cjk_character(ch):
    """cjk类字符(包括中文字符也在此列)
    参考: https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    C: chinese, J: japanese, K: korean
    0x4E00 <= code <= 0x9FFF, CJK Unified Ideographs, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    0x3400 <= code <= 0x4DBF, CJK Unified Ideographs Extension A, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A
    0x20000 <= code <= 0x2A6DF, CJK Unified Ideographs Extension B, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B
    0x2A700 <= code <= 0x2B73F, CJK Unified Ideographs Extension C, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_C
    0x2B740 <= code <= 0x2B81F, CJK Unified Ideographs Extension D, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_D
    0x2B820 <= code <= 0x2CEAF, CJK Unified Ideographs Extension E, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_E
    0x2CEB0 <= code <= 0x2EBEF, CJK Unified Ideographs Extension F, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_F
    0xF900 <= code <= 0xFADF, 兼容汉字
    0x2F800 <= code <= 0x2FA1F, 兼容扩展
    rference: https://www.cnblogs.com/straybirds/p/6392306.html
    """

    # The ord() function returns an integer representing the Unicode character.
    # by the way, the ord function is the inverse of chr()
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0x2CEB0 <= code <= 0x2EBEF or \
            0xF900 <= code <= 0xFADF or \
            0x2F800 <= code <= 0x2FA1F
```

**GB2312**编码：1981年5月1日发布的简体中文汉字编码国家标准。GB2312对汉字采用双字节编码，收录7445个图形字符，其中包括6763个汉字。

**BIG5**编码：台湾地区繁体中文标准字符集，采用双字节编码，共收录13053个中文字，1984年实施。

**GBK**编码：1995年12月发布的汉字编码国家标准，是对GB2312编码的扩充，对汉字采用双字节编码。GBK字符集共收录21003个汉字，包含国家标准GB13000-1中的全部中日韩汉字，和BIG5编码中的所有汉字。

**GB18030**编码：2000年3月17日发布的汉字编码国家标准，是对GBK编码的扩充，覆盖中文、日文、朝鲜语和中国少数民族文字，其中收录27484个汉字。GB18030字符集采用单字节、双字节和四字节三种方式对字符编码。兼容GBK和GB2312字符集。

**Unicode**编码：国际标准字符集，它将世界各种语言的每个字符定义一个唯一的编码，以满足跨语言、跨平台的文本信息转换。

 

汉子unicode编码表：

一般使用2w基本汉子就够了

|字符集|字数|Unicode 编码|
|-|-|-|
|基本汉字|20902字|4E00-9FA5|
|基本汉字补充|38字|9FA6-9FCB|
|扩展A|6582字|3400-4DB5|
|扩展B|42711字|20000-2A6D6|
|扩展C|4149字|2A700-2B734|
|扩展D|222字|2B740-2B81D|
|康熙部首|214字|2F00-2FD5
|部首扩展|115字|2E80-2EF3
|兼容汉字|477字|F900-FAD9
|兼容扩展|542字|2F800-2FA1D
|PUA(GBK)部件|81字|E815-E86F
|部件扩展|452字|E400-E5E8
|PUA增补|207字|E600-E6CF
|汉字笔画|36字|31C0-31E3
|汉字结构|12字|2FF0-2FFB
|汉语注音|22字|3105-3120
|注音扩展|22字|31A0-31BA
|〇|1字|3007|

keras.utils.to_categorical(y, num_classes=None, dtype='float32'):


```python
def to_categorical(y, num_classes=None, dtype='float32'):

    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
```

data_generator():

```python
# two sentences -> list of two lists of token id with label
# text_a: '工商银行是一家银行' -> x1 = ['token1', 'token2', ...]
# text_b: '工商银行'          -> x2 =['token1', 'token2', ...]
# label: '无'                -> y = np.array([1, 0])
```

early_stoping = keras.callbacks.EarlyStopping(monitor='accuracy',
                                              patience=5)

```python
# 早停法,防止过拟合,在监视的数值停止增加的时候停止训练
# early_stoping里有很多函数,并且这个对象能返回一个值,我有点奇怪为什么那个返回值的函数被调用了
# 我明白了,因为on_train_begin和on_epoch_end可以在特定时间被调用,有点类似于Magic method?
```