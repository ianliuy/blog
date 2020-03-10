# pandas中concat()使用

Concatenate pandas objects along a particular axis with optional set logic along the other axis.

Can also add layers of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.

**Parameters**

\-\-\-\-\-\-\-\-\-\-

obj: a sequence or mapping of Series / DataFrame

axis: 连接的轴向

join: 连接方式，类似数据库inner (default)和outer

ignore_index: 是否忽略原df的index。This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information. 

keys: 指定可以形成hierarchial index

levels: 没用过，不懂。（用于构造多索引的特定级别（唯一值）。否则它们将从键中推断出来。Specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred from the keys.）

verify_integrity: 没用过，不懂。（Check whether the new concatenated axis contains duplicates. This can be very expensive relative to the actual data concatenation.）

sort: join为outer时如果有未连接的没对齐连接轴就将其排序

copy: 不懂，什么叫do not copy data unnecessarily

**Returns**

\-\-\-\-\-\-\-\-\-\-

object. 这个object的类型根据传入的类型。如果两个Series，返回也是Series。如果传入至少有一个DF，返回DF。如果轴向是column (axis = 1)


**Examples**

\-\-\-\-\-\-\-\-\-\-

```python
>>> s1 = pd.Series(['a', 'b'])
>>> s2 = pd.Series(['c', 'd'])
>>> pd.concat(s1, s2)
0    a
1    b
0    c
1    d
# 不忽略index连接两个Series
```

```python
>>> pd.concat(s1, s2, ignore_index = True)
0    a
1    b
2    c
3    d
```

```python
>>> pd.concat([s1, s2], keys = ['s1', 's2'])
s1  0    a
    1    b
s2  0    c
    1    d
# hierarchical index
```

```python
>>> df1 = pd.DataFrame([['a', '1'], ['b', '2']], 
···                    columns = ['letter', 'number'])
letter number
a      1
b      2
>>> df3 = pd.DataFrame([['c', '3', 'cat'], ['d', '4', 'dog']], 
···                    columns = ['letter', 'number', 'animal'])
letter numbr animal
c      3     cat
d      4     dog
>>> pd.concat([df1, df3])
    letter number animal
0   a      1      NaN
1   b      2      NaN
0   c      3      cat
1   d      4      dog
# pd.concat([df1, df3]) 
# equals to 
# pd.concat([df1, df3], *axis = 0*, *join = 'outer'*, *sort = False*)
```
