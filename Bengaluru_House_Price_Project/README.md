
- [Introduction 前言](#introduction-前言)
- [Software and environment 软件和安装环境](#software-and-environment-软件和安装环境)
  - [Software](#software)
  - [Language](#language)
  - [Folder](#folder)
  - [Virtual environment](#virtual-environment)
    - [MacOS: Terminal](#macos-terminal)
    - [VS Code](#vs-code)
    - [Flask](#flask)
    - [Postman](#postman)
- [Model 模型](#model-模型)
  - [Data Cleaning](#data-cleaning)
    - [进行数据清洗时，先了解数据集里面有什么feature](#进行数据清洗时先了解数据集里面有什么feature)
    - [查看认为不需要的数据集并且drop掉](#查看认为不需要的数据集并且drop掉)
    - [drop 掉null value](#drop-掉null-value)
    - [查看不一致的单位，并且进行更换。](#查看不一致的单位并且进行更换)
    - [转换小数](#转换小数)
  - [Feature Engineering](#feature-engineering)
    - [改变feature](#改变feature)
    - [移除字符串头尾指定的字符](#移除字符串头尾指定的字符)
    - [把同类组合在一起/ 方便之后降维度](#把同类组合在一起-方便之后降维度)
    - [移除outliner](#移除outliner)
      - [remove through business logics](#remove-through-business-logics)
      - [remove outlier through standard deviation and mean](#remove-outlier-through-standard-deviation-and-mean)
      - [remove outlier through data visualization](#remove-outlier-through-data-visualization)
      - [outlier removal using Bathrooms feature](#outlier-removal-using-bathrooms-feature)
  - [Model Building](#model-building)
    - [One Hot Encoding for Location](#one-hot-encoding-for-location)
    - [sklearn](#sklearn)
      - [使用LinearRegression](#使用linearregression)
      - [score](#score)
      - [optimization GridSearchCV](#optimization-gridsearchcv)
      - [测试](#测试)
  - [Packaging](#packaging)
    - [pickle](#pickle)
    - [json](#json)
- [Server 伺服器](#server-伺服器)
  - [Flask](#flask-1)
  - [Port 报错](#port-报错)
  - [Server.py](#serverpy)
  - [util.py](#utilpy)
- [Client 用户界面](#client-用户界面)
  - [app.html](#apphtml)
  - [app.css](#appcss)
  - [app.js](#appjs)
- [Final product 成品](#final-product-成品)
  - [User Interface 用户使用界面](#user-interface-用户使用界面)
  - [get_location_names](#get_location_names)
  - [predict_home_price](#predict_home_price)
- [Summary 反思](#summary-反思)
- [Source 练习来源](#source-练习来源)
- [Reference 参考资料](#reference-参考资料)
# Introduction 前言 
本文希望透过练习[codebasics](https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices) 提供的`banglorehomeprices`题目，实践在[matplotlib, numpy, pandas 学习笔记](https://blog.csdn.net/m0_66706847/article/details/126229590?spm=1001.2014.3001.5501)中学到的知识。

主要内容分为六个部分： 
1. [Software and environment 软件和安装环境](#Software_and_environment)
2. [Model 模型](#Model)
3. [Server 伺服器](#Server)
4. [Client 用户界面](#Client)
5. [Final product 成品](#Final_product)
6. [Summary 反思](#Summary)

# Software and environment 软件和安装环境
<span id='Software_and_environment'></span>

## Software
Data Science: Jupyter Notebook
Web Development: Visual Studio Code/ PyCharm + Flask + Postman

## Language 
Python + HTML5, CSS, Javascript

## Folder 
Pickle: Python中的Pickle主要用于序列化和反序列化一个Python对象结构。换句话说，它是将Python对象转换为字节流的过程，以便将其存储在文件/数据库中，保持跨会话的程序状态，或者通过网络传输数据。


Json: 

## Virtual environment 
- 虚拟环境是一种工具，它通过为不同的项目创建隔离的Python虚拟环境来帮助保持不同项目所需的依赖关系。
- virtualenv 避免了全局安装 Python 包的需要。当 virtualenv 被激活时，pip 将在环境中安装软件包，这不会以任何方式影响基本的 Python 安装。
- 使用系统中的 Python 和库运行时，只能使用一个特定的 Python 版本，试图在一个 Python 安装上运行所有的 Python 应用程序，很可能会在库的集合中发生版本冲突，也有可能对系统 Python 的改变会破坏其他依赖它的操作系统功能。

### MacOS: Terminal
```bash
virtualenv codebasics -p python3
source codebasics/bin/activate
# 进入建造的环境
(codebasics) (base) haleyk@Kwoks-MacBook-Air ~ %
# 下載package 
pip3 install flask
...
# 檢查package
python3
>>> import flask
# 沒有反應證明安裝成功
# 退出
>>> exit()
```

### VS Code
输入跟virtual environment的关键词`codebasics`就能在VS Code上当它当interpreter跑程序：

![请添加图片描述](https://img-blog.csdnimg.cn/5f50087f14c34b90b7531d0075f314ca.png)

### Flask
Flask是一个用Python编写的微型网络框架。它被归类为一个微框架，因为它不需要特定的工具或库。

                                                                  
### Postman
Postman是一个API平台，供开发者设计、构建、测试和迭代他们的API。
设置`GET`，输入port `http://127.0.0.1:5000/get_location_names`后，点击`Send`；下方的界面会显示页面要返回的内容：
![请添加图片描述](https://img-blog.csdnimg.cn/ac6328466da748a9b2c7263e724b18a0.png)


---
# Model 模型
<span id='Model'></span>

使用Bengaluru_House_Data.csv 数据集进行[Data Cleaning](#Data_cleaning)、[Feature Engineering](#Feature_engineering)、[Model Building](#Model_building) 和 [Packaging](#Packaging)，打造可视化的数据集。


## Data Cleaning
<span id='Data_cleaning'></span>

### 进行数据清洗时，先了解数据集里面有什么feature

```python
df1.shape
df1

# return
```
![请添加图片描述](https://img-blog.csdnimg.cn/96b28e1c6c4b4babaefde31951b43200.png)

### 查看认为不需要的数据集并且drop掉

```python
df1.groupby('area_type')['area_type'].count() # agg('count')

# return
area_type
Built-up  Area          2418
Carpet  Area              87
Plot  Area              2025
Super built-up  Area    8790
Name: area_type, dtype: int64
```

```python
# Drop features that are not required to build our model
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
df2.head()
```

### drop 掉null value

```python
df2.isnull().sum()

# return
location       1
size          16
total_sqft     0
bath          73
price          0
dtype: int64
```

```python
df3 = df2.dropna()
df3.isnull().sum()
```

### 查看不一致的单位，并且进行更换。

```python
df3['size'].unique()

# return
array(['2 BHK', '4 Bedroom', '3 BHK', '4 BHK', '6 Bedroom', '3 Bedroom',
       '1 BHK', '1 RK', '1 Bedroom', '8 Bedroom', '2 Bedroom',
       '7 Bedroom', '5 BHK', '7 BHK', '6 BHK', '5 Bedroom', '11 BHK',
       '9 BHK', '9 Bedroom', '27 BHK', '10 Bedroom', '11 Bedroom',
       '10 BHK', '19 BHK', '16 BHK', '43 Bedroom', '14 BHK', '8 BHK',
       '12 Bedroom', '13 BHK', '18 Bedroom'], dtype=object)
```

```python
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0])) # 拆解' '，取第一个元素

# return 即为 2, 4 ....

df3['bhk'].unique()

# return
array([ 2,  4,  3,  6,  1,  8,  7,  5, 11,  9, 27, 10, 19, 16, 43, 14, 12,
       13, 18])
```
### 转换小数

```python
df3.total_sqft.unique()
# return
array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],
      dtype=object)
      
# Explore total_sqft feature
def is_float(x):
        try:
                float(x)
                return True
        except ValueError:
                return False

df3[df3['total_sqft'].apply(is_float)].head(10) # 没有反应，因为有' - '

def convert_sqft_to_num(x):
        token = x.split('-')
        if len(token) == 2:
                return (float(token[0]) + float(token[1])) / 2
        try:
            return float(x)
        except:
            return np.nan

convert_sqft_to_num('1000')
# return
1000.0

convert_sqft_to_num('1000-1200')
# return
1100.0

convert_sqft_to_num('34.465q. Meter')
# return
nan
```

```python
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()

# return
location	size	total_sqft	bath	price	bhk
0	Electronic City Phase II	2 BHK	1056.0	2.0	39.07	2
1	Chikka Tirupathi	4 Bedroom	2600.0	5.0	120.00	4
2	Uttarahalli	3 BHK	1440.0	2.0	62.00	3
3	Lingadheeranahalli	3 BHK	1521.0	3.0	95.00	3
4	Kothanur	2 BHK	1200.0	2.0	51.00	2
```


## Feature Engineering
<span id='Feature_engineering'></span>

### 改变feature

```python
# Add new feature called price per square feet
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000 / df5['total_sqft']
df5.head()

# return
location	size	total_sqft	bath	price	bhk	price_per_sqft
0	Electronic City Phase II	2 BHK	1056.0	2.0	39.07	2	3699.810606
1	Chikka Tirupathi	4 Bedroom	2600.0	5.0	120.00	4	4615.384615
2	Uttarahalli	3 BHK	1440.0	2.0	62.00	3	4305.555556
3	Lingadheeranahalli	3 BHK	1521.0	3.0	95.00	3	6245.890861
4	Kothanur	2 BHK	1200.0	2.0	51.00	2	4250.000000
```
### 移除字符串头尾指定的字符


```python
df5.location

# return
0        Electronic City Phase II
1                Chikka Tirupathi
2                     Uttarahalli
3              Lingadheeranahalli
4                        Kothanur
                   ...           
13315                  Whitefield
13316               Richards Town
13317       Raja Rajeshwari Nagar
13318             Padmanabhanagar
13319                Doddathoguru
Name: location, Length: 13246, dtype: object

# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations
df5.location = df5.location.apply(lambda x: x.strip()) # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列

location_stats = df5.groupby('location')['location'].count()  # how many types does it have
location_stats

# return
location
1 Annasandrapalya                                  1
1 Giri Nagar                                       1
1 Immadihalli                                      1
1 Ramamurthy Nagar                                 1
12th cross srinivas nagar banshankari 3rd stage    1
                                                  ..
t.c palya                                          1
tc.palya                                           4
vinayakanagar                                      1
white field,kadugodi                               1
whitefiled                                         1
Name: location, Length: 1293, dtype: int64
```


### 把同类组合在一起/ 方便之后降维度

```python
len(df5['location'].unique()) # it shows that we have too much location data

# return
1293

# Any location having less than 10 data points should be tagged as "other" location. 
# This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, 
# it will help us with having fewer dummy columns
location_stats_less_than_10 = location_stats[location_stats <= 10] # get the locations with less than 10

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique()) # it shows that we have reduced the location data

# return 
242
# use the one-hot encoding to reduce the dimensionality of the data afterwards, there is only 242 number of data to concern with
```
### 移除outliner 
#### remove through business logics 
通常每间卧室的面积是300（即一个两居室的单位至少是600平方英尺。如果两居室单位有400平方英尺，那么这似乎很可疑，可以作为一个离群点被删除。我们将通过保持每个单位300平方英尺的最低售价来消除这个离群值。

```python
df5[df5.total_sqft/df5.bhk<300].head() 

# return
	location	size	total_sqft	bath	price	bhk	price_per_sqft
9	other	6 Bedroom	1020.0	6.0	370.0	6	36274.509804
45	HSR Layout	8 Bedroom	600.0	9.0	200.0	8	33333.333333
58	Murugeshpalya	6 Bedroom	1407.0	4.0	150.0	6	10660.980810
68	Devarachikkanahalli	8 Bedroom	1350.0	7.0	85.0	8	6296.296296
70	other	3 Bedroom	500.0	3.0	100.0	3	20000.000000


df6 = df5[~(df5.total_sqft/df5.bhk<300)] # negate on the criteria if you want to filter the rows
df6.shape
```

#### remove outlier through standard deviation and mean


```python
df6.price_per_sqft.describe()

# Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, 
# this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation

# return
count     12456.000000
mean       6308.502826
std        4168.127339
min         267.829813
25%        4210.526316
50%        5294.117647
75%        6916.666667
max      176470.588235
Name: price_per_sqft, dtype: float64

def remove_pps_outliners(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (mean - std)) & (subdf.price_per_sqft < (mean + std))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliners(df6)
df7.shape
```

#### remove outlier through data visualization

remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

```python
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    # matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color='blue', label='2 BHK')
    plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, marker = '+', label='3 BHK')
    plt.xlabel('Total Square Feet')
    plt.ylabel('Price Per Square Feet')
    plt.title('location')
    plt.legend()

plot_scatter_chart(df7, 'Rajaji Nagar')
```

```python
def remove_bhk_outliners(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')

df8 = remove_bhk_outliners(df7)
df8.shape
```

#### outlier removal using Bathrooms feature
4 bedroom home and even if bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. 
Anything above that is an outlier or a data error and can be removed
```python
df8[df8.bath>df8.bhk+2]


df9 = df8[df8.bath < df8.bhk+2]
df9.shape
```

---


## Model Building
<span id='Model_building'></span>

```python
df10 = df9.drop(['price_per_sqft', 'size'], axis = 1)
df10.head()
```


### One Hot Encoding for Location
后续drop location，因为已经转换成dummies 了
```python
dummies = pd.get_dummies(df10.location) # one-hot encoding
dummies.head()

# return
1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	5th Phase JP Nagar	6th Phase JP Nagar	7th Phase JP Nagar	8th Phase JP Nagar	9th Phase JP Nagar	...	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur	other
0	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0


```

other 是不重要的：
```python
df11 = pd.concat([df10, dummies.drop('other', axis = 1)], axis = 1)
df11

# return
	location	total_sqft	bath	price	bhk	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	1st Block Jayanagar	2850.0	4.0	428.0	4	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1st Block Jayanagar	1630.0	3.0	194.0	3	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1st Block Jayanagar	1875.0	2.0	235.0	3	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1st Block Jayanagar	1200.0	2.0	130.0	3	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1st Block Jayanagar	1235.0	2.0	148.0	2	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
10232	other	1200.0	2.0	70.0	2	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10233	other	1800.0	1.0	200.0	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10236	other	1353.0	2.0	110.0	2	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10237	other	812.0	1.0	26.0	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10240	other	3600.0	5.0	400.0	4	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
```

drop location：
因为已经转换成dummies 了

```python
df12 = df11.drop('location', axis = 1)
df12.head()
```

用price对比房子：
X 是剩下的feature
```python
X = df12.drop('price', axis = 1)
X.head()

# return
total_sqft	bath	bhk	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	5th Phase JP Nagar	6th Phase JP Nagar	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	2850.0	4.0	4	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1630.0	3.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1875.0	2.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1200.0	2.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1235.0	2.0	2	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
```

y 是房子的價格
```python
y = df12.price
y.head()
```

### sklearn

#### 使用LinearRegression
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# return
0.845227769787433
```

#### score
Use K Fold cross validation to measure accuracy of our LinearRegression model

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
cross_val_score(lr, X, y, cv = cv)

# return
array([0.82430186, 0.77166234, 0.85089567, 0.80837764, 0.83653286])
```

#### optimization GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_the_best_model_using_gridssearchcv(X, y):
    algorithms = {
        'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'normalize': [True, False]
        }
        },
        'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [1, 2],
            'selection': ['random', 'cyclic']
        }
        },
        'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random']
        }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    for algo_name, config in algorithms.items():
       gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score = True)
       gs.fit(X, y)
       scores.append({
        'model': algo_name, 
        'best_score': gs.best_score_,
        'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])

find_the_best_model_using_gridssearchcv(X, y)

# return
model	best_score	best_params
0	linear_regression	0.818354	{'normalize': False}
1	lasso	0.687459	{'alpha': 2, 'selection': 'random'}
2	decision_tree	0.737432	{'criterion': 'friedman_mse', 'splitter': 'ran...
```



#### 测试

```python
X.columns
Index(['total_sqft', 'bath', 'bhk', '1st Block Jayanagar',
       '1st Phase JP Nagar', '2nd Phase Judicial Layout',
       '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar',
       '6th Phase JP Nagar',
       ...
       'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout',
       'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka',
       'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur'],
      dtype='object', length=244)
      
np.where(X.columns == '2nd Phase Judicial Layout')[0][0] # index of 2nd Phase Judicial Layout
# if one [0] only, return array([5])

# return 
5
```

```python

def predict_price(location, sqft, bhk, bath):
    loc_index = np.where(X.columns == location)[0][0] # index of location

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1 
    return lr.predict([x])[0]

predict_price('1st Phase JP Nagar', 1000, 2, 2)
# return
83.49904677187278
```

---

## Packaging 
<span id='Packaging'></span>

### pickle 

```python
# Export the tested model to a pickle file

import pickle 
with open('banglore_home_prices_model.pickle', 'wb') as f: # 命名为banglore_home_prices_model.pickle
    pickle.dump(lr, f) # linearregression model 
    
# 'wb' means 'write binary' and is used for the file handle: open('save. p', 'wb' ) which writes the pickeled data into a file    
# To pickle an object into a file, call pickle.dump(object, file). To get just the pickled bytes, call pickle.dumps(object).
```


### json
```python
# Export location and column information to a file that will be useful later on in prediction application
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f: # 命名为columns.json
    f.write(json.dumps(columns)) # dump 将 Python 对象编码成 JSON 字符串，并写入文件中
```
![请添加图片描述](https://img-blog.csdnimg.cn/857c62101c3c4c8d811c147ffb6d80ed.png)


---
# Server 伺服器
<span id='Server'></span>
![请添加图片描述](https://img-blog.csdnimg.cn/66b922e0595f4277830a6b8f03c1ff6c.png)

## Flask 

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello World!"

if __name__ ==  "__main__":
    print("Start Python Flask Server For Banglore Home Price Prediction.")
    app.run()
```

```python
# return
 * Running on http://127.0.0.1:5000
```

直接点击会报错：
![请添加图片描述](https://img-blog.csdnimg.cn/f6f672c68ef5454eb97e6e331a277b97.png)
在port后面需要输入想要打开的route：
```python
http://127.0.0.1:5000/hello
```

页面会显示`hello()`函数返回的内容，即为`Hello World!`


## Port 报错
若果返回错误（通常是macOS 版本的Monterey）， localhost 5000 端口被一个叫 AirPlay Receiver 的服务占用了。Flask 内置服务器默认运行在 5000 端口，造成端口冲突。当你通过将 host 设为 0.0.0.0 指定 Flask 的内置服务器对外可见，或是使用内网穿透工具时，会发现程序无法访问（有时未设置对外可见也会遇到这个问题）：

```python
Address already in use
Port 5000 is in use by another program. Either identify and stop that program, or start the server with a different port.
On macOS, try disabling the 'AirPlay Receiver' service from System Preferences -> Sharing.
```

解决方法：
方法一（推荐）： 系统设置（System Preferences） > 分享（Sharing） > AirPlay Receiver > 取消勾选
![请添加图片描述](https://img-blog.csdnimg.cn/cfbe8a94c25045e780d5e42ee14c51d1.png)
*可能需要重启电脑

方法二： 更改 Flask 开发服务器默认的端口。在执行 flask run 命令时使用 -p/–port 选项可以自定义端口
```bash
$ flask run -p 4999  # 或 flask run --port 4999
```

方法三：在执行 flask run 之前通过环境变量 FLASK_RUN_PORT 设置：
```bash
$ export FLASK_RUN_PORT=8000 # macOS/Linux
```


## Server.py
```python
from flask import Flask, request, jsonify
import util

app = Flask(__name__) 
'''
The variable __name__ is passed as first argument when creating an instance of the Flask object (a Python Flask application). 
In this case __name__ represents the name of the application package and it's used by Flask to identify resources
'''

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


# making HTTP call from HTML application
@app.route('/predict_home_price', methods=['GET','POST']) # method to be commissioned
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts() # load the saved artifacts and open the files in util.py
    app.run()
```

## util.py 

```python
import json 
import pickle 
import numpy as np

# global variables
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower()) 
    except:
        loc_index = -1 # if location is not found, return -1
         
    x = np.zeros(len(__data_columns)) # create a numpy array of zeros of size len(__data_columns)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2) # return only one element from the saved artifacts with 2 decimal numbers

'''

predict price model from Project.ipynb
def predict_price(location, sqft, bhk, bath):
    loc_index = np.where(X.columns == location)[0][0] 

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1 
    return lr.predict([x])[0]

'''
def load_saved_artifacts():
    print("loading saved artifacts...\n")
    global __data_columns
    global __locations

    # open the file
    with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:] # first 3 columns are "total_sqft", "bath", "bhk", and [3:] starts with "location" 

    global __model
    if __model is None:
        with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done\n")
    # with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
    #     __model = pickle.load(f) -> need global variable __model to be able to use it in the predict_price function
    # print("loading saved artifacts...done\n")

# def get_data_columns():
#     return __data_columns

def get_location_names():
    return __locations

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    # for testing
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 2))
    print(get_estimated_price('Kalhalli', 1000, 3, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 3, 2)) # other location
```

---
# Client 用户界面
<span id='Client'></span>

## app.html

```html
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script> <!-- open-source JavaScript libraries -->
    <script src="app.js"></script> <!-- javascript -->
	<link rel="stylesheet" href="app.css"> <!-- css -->
```

## app.css
在html 上用`class`：

```html
<select class="location" name="" id="uiLocations">
```
在css 用 `.location` 对照：

```css
.location{
  font-family: "Roboto", sans-serif;
  outline: 0;
  background: #f2f2f2;
  width: 76%;
  border: 0;
  margin: 0 0 10px;
  padding: 10px;
  box-sizing: border-box;
  font-size: 15px;
  height: 40px;
  border-radius: 5px;
}
```

## app.js

url linkage 
![请添加图片描述](https://img-blog.csdnimg.cn/3a7fa67a05844b9daab017d648886bb7.png)





---
# Final product 成品
<span id='Final_product'></span>

## User Interface 用户使用界面
![请添加图片描述](https://img-blog.csdnimg.cn/02b7e130421442feb9ac49bc034c9a58.png)
## get_location_names
![请添加图片描述](https://img-blog.csdnimg.cn/3b25e0c1132c4d7f9d864ff471dafb62.png#pic_center)

## predict_home_price
![请添加图片描述](https://img-blog.csdnimg.cn/b6268d44fa2d4e4aa7301b6b04285e7b.png#pic_center)

---
# Summary 反思
<span id='Summary'></span>

1. datascience: 需要熟悉数据集的feature才能进行筛选；多练习；还有补machine learning的知识
2. web development: flask, postman 等工具；需要补web方面的知识，前端三剑客、还有框架
3. datascience和web development之间的联系，可以说是backend 和 frontend 吧

持续学习，无限进步哈哈哈

---
# Source 练习来源
[Github: codebasics_datascience_BangloreHomePrices](https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices)

---
# Reference 参考资料

[virtualenv](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one)
[如果你在 macOS 上无法访问 Flask 程序](https://greyli.com/thank-you-apple/)