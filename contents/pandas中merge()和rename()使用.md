# pandas中merge()和rename()使用

## pandas.DataFrame.merge()使用

reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html

DataFrame.merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None) → 'DataFrame'

“join columns with other dataframe either on index or on a key column”

**Parameters**

\-\-\-\-\-\-\-\-\-\-

other：另一个要连接的dataframe/ named series

how: 指定连接的方法，包括左连接left，右连接right，内连接inner， 外连接outer

on：两者都有的列，用于连接

left_on：左侧DataFrame中有的列，用于连接

right_on：右侧DataFrame中的列，用于连接

left_index：是否用左侧DataFrame的index连接

right_index：是否用右侧DataFrame的index连接

sort：是否按照字母顺序排序。如果否，merge后的顺序取决于how

suffixes：str的tuple，两个df同名列的后缀

copy：default Ture

indicator：不懂

validate：“1:1”检查是否一一对应, "1:m"检查左边一一对应, "m:1"检查右边一一对一, "m:m"可以用，但不检查

**Returns**

\-\-\-\-\-\-\-\-\-\-

DataFrame

A dataframe of the two merged objects

## pandas.DataFrame.rename()使用

reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html

DataFrame.rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')

"Alter axis labels"

**Parameters**

\-\-\-\-\-\-\-\-\-\-

mapper: 类字典结构或函数，与axis配合指定修改的列还是行

index: columns: 可替代的指定轴。其中 mapper, axis = 1等同于 columns = mapper

axis: ("index", "columns")或(0, 1)

copy: 默认true

inplace: 是否生成新的DataFrame

level: 跟multiIndex相关的

errors: 默认ignore，但是也可以raise (KeyError)

**Returns**

\-\-\-\-\-\-\-\-\-\-

DataFrame

DataFrame with the renamed axis labels

**Raises**

\-\-\-\-\-\-\-\-\-\-

KeyError

如果error == 'raise' 并且选定轴上有未找到的label

