# Roberta-Keras-Tutorials

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