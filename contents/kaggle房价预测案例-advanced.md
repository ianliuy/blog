```python
import numpy as np
import pandas as pd # 变成类似数据库的形式存储
```


```python
# 1. 检查原数据集，看一下数据集长什么样
train_df = pd.read_csv('./input/train.csv', index_col = 0)
test_df = pd.read_csv('./input/test.csv', index_col = 0)
```


```python
train_df.head()
# 英文词 没有强烈地数学意义
# 数学词 比如index 没有强烈的含义
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
# 合并数据，将SalePrice拿出来 作为训练目标 只出现在训练集中 不出现在测试集中
# SalePrice这个分布不是类正态的分布，需要纠正一下（正则化）
%matplotlib inline
price = pd.DataFrame({'price': train_df['SalePrice'], 'log(price + 1)': np.log1p(train_df['SalePrice'])})
price.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002676967DA58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000267695D3630>]],
          dtype=object)




![output_3_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7edn228j30af07c0sn.jpg)



```python
# 1. 预处理Y（这里用了log比较简单粗暴，别的也可以）
# 怎么来的就要怎么回去，这里用到了log1p，回去就是expm1
# 同理，log()的话就要exp() etc
y_train = np.log1p(train_df.pop('SalePrice'))
```


```python
all_df = pd.concat([train_df, test_df], axis = 0)
```


```python
all_df.shape
```




    (2919, 79)




```python
all_df.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2910</th>
      <td>180</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1470</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2911</th>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1484</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2912</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>13384</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2913</th>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1533</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>2914</th>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1526</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2915</th>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1936</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2916</th>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1894</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>2917</th>
      <td>20</td>
      <td>RL</td>
      <td>160.0</td>
      <td>20000</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2919</th>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 79 columns</p>
</div>




```python
# 变量变化（特征工程）
# 不方便处理或不unify数据都数据化表示
# 每种变量都没有特定的数据处理方式，需要自己去想
all_df['MSSubClass'].dtypes
```




    dtype('int64')




```python
all_df['MSSubClass'] = all_df['MSSubClass'].astype('str')
```


```python
all_df['MSSubClass'].value_counts()
```




    20     1079
    60      575
    50      287
    120     182
    30      139
    160     128
    70      128
    80      118
    90      109
    190      61
    85       48
    75       23
    45       18
    180      17
    40        6
    150       1
    Name: MSSubClass, dtype: int64




```python
# 把categorical的变量变为numerical
# 使用的方法是one-hot独热码
pd.get_dummies(all_df['MSSubClass'], prefix = 'MSSubClass').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass_120</th>
      <th>MSSubClass_150</th>
      <th>MSSubClass_160</th>
      <th>MSSubClass_180</th>
      <th>MSSubClass_190</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_30</th>
      <th>MSSubClass_40</th>
      <th>MSSubClass_45</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_70</th>
      <th>MSSubClass_75</th>
      <th>MSSubClass_80</th>
      <th>MSSubClass_85</th>
      <th>MSSubClass_90</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_dummy_df = pd.get_dummies(all_df)
all_dummy_df.shape
```




    (2919, 303)




```python
# 处理numerical变量：数据缺失
all_dummy_df.isnull().sum().sort_values(ascending = False).head(15)
```




    LotFrontage             486
    GarageYrBlt             159
    MasVnrArea               23
    BsmtHalfBath              2
    BsmtFullBath              2
    BsmtFinSF2                1
    GarageCars                1
    TotalBsmtSF               1
    BsmtUnfSF                 1
    GarageArea                1
    BsmtFinSF1                1
    Condition1_Artery         0
    Condition2_Feedr          0
    Condition2_Artery         0
    Neighborhood_Somerst      0
    dtype: int64




```python
# 一般处理缺失，先看数据集说明“这些缺失代表什么”
# 如果数据集没说，就要靠自己判断
# 如果缺失很少，可以靠补充平均值
# 如果缺失很多，就可以考虑删掉这一列
mean_cols = all_dummy_df.mean()
mean_cols.head(10)
```




    LotFrontage        69.305795
    LotArea         10168.114080
    OverallQual         6.089072
    OverallCond         5.564577
    YearBuilt        1971.312778
    YearRemodAdd     1984.264474
    MasVnrArea        102.201312
    BsmtFinSF1        441.423235
    BsmtFinSF2         49.582248
    BsmtUnfSF         560.772104
    dtype: float64




```python
all_dummy_df = all_dummy_df.fillna(mean_cols)
```


```python
all_dummy_df.isnull().sum().sum()
```




    0




```python
# 标准化numerical数据（optional）
# regression分类器，最好把数据放在一个区间内
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols
```




    Index(['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
           'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
           'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
           'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
           'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
           'MoSold', 'YrSold'],
          dtype='object')




```python
# 计算标准分布(X - X')/s
# 使数据平滑
# 多种方法
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std 
```


```python
# 经过上次，把数据清理干净，平滑数据
# 划分测试集和训练集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
```


```python
dummy_train_df.shape, dummy_test_df.shape
```




    ((1460, 303), (1459, 303))




```python
# 第一步，先用Ridge Regression模型跑跑看
# RG模型的好处是可以把所有的variable都放过去
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
```


```python
# 把所有的df变成numpy array， 这一步不是很必要，但是与sklearn更加贴合
X_train = dummy_train_df.values
X_test = dummy_test_df.values
```

# 从这里开始不同


```python
# 一般来说，单个分类器的效果非常有限，
# 会倾向于把N多个分类器合在一起，
# 做一个“综合分类器”，达到最好的效果
# 从刚刚结果得知，Ridge(alpha = 15)给了最好的效果
from sklearn.linear_model import Ridge
ridge = Ridge(15)
```


```python
# bagging把很多小分类器放在一起，每个train随机的一部分数据，然后把结果综合起来
# sklearn直接提供了框架，直接调用就好
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
```


```python
params = list(range(10, 21))
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators = param, base_estimator = ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")
```




    Text(0.5, 1.0, 'n_estimator vs CV Error')




![output_27_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7f0sibpj30av07c3yn.jpg)



```python
# 用20个分类器，达到了最好版本0.132
# 之前，用一个分类器，只达到了0.135
# 如果没有用ridge，也可以用sklearn默认的DecisionTree
params = list(range(10, 100, 10))
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")
# 结果发现不太行，还不如单个Ridge分类效果好
```




    Text(0.5, 1.0, 'n_estimator vs CV Error')




![output_29_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7f7pyk3j30ao07cdfv.jpg)



```python
# Boosting
# Boosting比Bagging更高级一点，也是揽来一把的分类器。
# 但是将他们线性排列，下一个分类器把上一个分类器分类得更不好的地方加上更高的权重
# 下一个分类器能在这个部分学得更好
from sklearn.ensemble import AdaBoostRegressor
```


```python
params = list(range(1,20))
test_scores = []
for param in params:
    clf = AdaBoostRegressor(n_estimators = param, base_estimator = ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")
# 效果并没有Ridge+Bagging好
# 不稳定的解决方法，在区间内再细分，同时减少cv folder5或3
```




    Text(0.5, 1.0, 'n_estimator vs CV Error')




![output_32_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7fgt435j30ao07cdfw.jpg)



```python
params = list(range(10,100, 10))
test_scores = []
for param in params:
    clf = AdaBoostRegressor(n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")
# 效果并没有Ridge+Bagging好
# 不稳定的解决方法，在区间内再细分，同时减少cv folder5或3
```




    Text(0.5, 1.0, 'n_estimator vs CV Error')




![output_33_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7fogu8bj30ao07c3yl.jpg)


# XGBoost


```python
# 也是一款boost模型，但是做了很多的改进
# 全称：extreme gradient boost
from xgboost import XGBRegressor
# 梯度1. 更加防止过拟合2. 结果更好
```


```python
params = list(range(1, 10))
test_scores = []
for param in params:
    clf = XGBRegressor(max_depth = 2)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")
# 这就是为什么大家都在用XGBoost
```




    Text(0.5, 1.0, 'n_estimator vs CV Error')




![output_36_1.png](https://tva1.sinaimg.cn/large/006xRaCrly1gdd7fxjbc6j30ao07cq30.jpg)



```python
print(test_scores)
```

    [0.13796838085599003, 0.12794400478720266, 0.13118904945323134, 0.13580094464821596, 0.13693068594805988, 0.1430566185581099, 0.14286139970158934, 0.14273019872560852, 0.14421470830911928]
    
