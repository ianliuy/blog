# pandas中reset_index()&isnull()使用
## pandas.DataFrame.reset_index

"reset the inde of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, the method can remove one or more levels."

**Parameters**

\-\-\-\-\-\-\-\-\-\-

level: MultiIndex有关的，我还不太懂。
int, str, tuple, or list, default None Only remove the given levels from the index. Removes all levels by default.

drop: default False, 是否不将原index作为'index'列插入DataFrame开头

inplace: 返回一个新object还是修改这个

col_level: 如果行有MultiIndex，那么就决定那种程度的插入。默认第一层

col_fill: 如果行有MultiIndex，决定其他层怎么被命名

**Returns**

\-\-\-\-\-\-\-\-\-\-

DataFrame

DataFrame with the new index or None if inplace = True

```python
Example
----
df = df.reset_index(drop = True)
```

## pandas.Series.isnull

Detecting missing value, like NaN. Characters like empty string '' or numpy.inf are not considered NA values unless intentionally set pandas.options.mode_use_inf_as_na = True

**Returns**

\-\-\-\-\-\-\-\-\-\-

Series

Mask of bool values for each elements in Series that indicates whether an element is not an NA value.

```python
df = df.loc[df.content.isnull(), ['sentiment', 'label']]
# 使用isnull()方法，得到'content'为空的行的mask
# 然后用.loc方法将mask行的['sentiment', 'label']取出
# 返回一个新的DataFrame（名字还是df）
```
