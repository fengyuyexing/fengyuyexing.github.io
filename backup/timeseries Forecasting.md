# [使用 LSTM 进行多变量时间序列预测的保姆级教程](https://avoid.overfit.cn/post/1a36216705f2441b80fca567ea61e365)


```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
```


```python
# 设置单元格多个输出
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
# 列出所有已安装的包及其版本
!pip freeze > requirements.txt
```


```python
# 使用 watermark 魔术命令（推荐）
# 安装 watermark 扩展后，可以一键输出环境信息：

%load_ext watermark
# 显示所有导入包的版本
%watermark -iv 
```

    seaborn   : 0.12.2
    IPython   : 8.12.2
    tensorflow: 2.3.1
    sklearn   : 1.0.2
    matplotlib: 3.5.1
    numpy     : 1.18.5
    pandas    : 1.4.1
    
    


```python

```

## 加载数据，并检查输出
* csv文件中包含了谷歌从2001-01-25到2021-09-29的股票数据，数据是按照天数频率的。

[如果您愿意，您可以将频率转换为“B”[工作日]或“D”，因为我们不会使用日期，我只是保持它的现状。]

这里我们试图预测“Open”列的未来值，因此“Open”是这里的目标列


```python
df=pd.read_csv("train.csv",parse_dates=["Date"],index_col=[0])
df.shape
```




    (5203, 5)




```python
df.head()
df.tail()
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-01-25</th>
      <td>356.730774</td>
      <td>362.980774</td>
      <td>352.403839</td>
      <td>353.365387</td>
      <td>197.122452</td>
    </tr>
    <tr>
      <th>2001-01-26</th>
      <td>357.211548</td>
      <td>360.096161</td>
      <td>342.788452</td>
      <td>343.269226</td>
      <td>191.490234</td>
    </tr>
    <tr>
      <th>2001-01-29</th>
      <td>345.153839</td>
      <td>355.769226</td>
      <td>338.461548</td>
      <td>341.384613</td>
      <td>190.439011</td>
    </tr>
    <tr>
      <th>2001-01-30</th>
      <td>344.307678</td>
      <td>355.923065</td>
      <td>341.692322</td>
      <td>355.769226</td>
      <td>198.463318</td>
    </tr>
    <tr>
      <th>2001-01-31</th>
      <td>359.615387</td>
      <td>361.153839</td>
      <td>350.461548</td>
      <td>353.692322</td>
      <td>197.304749</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-09-23</th>
      <td>99.529999</td>
      <td>104.080002</td>
      <td>99.519997</td>
      <td>102.959999</td>
      <td>102.789993</td>
    </tr>
    <tr>
      <th>2021-09-24</th>
      <td>102.660004</td>
      <td>104.199997</td>
      <td>102.599998</td>
      <td>103.800003</td>
      <td>103.709198</td>
    </tr>
    <tr>
      <th>2021-09-27</th>
      <td>104.550003</td>
      <td>106.330002</td>
      <td>104.389999</td>
      <td>105.349998</td>
      <td>105.257835</td>
    </tr>
    <tr>
      <th>2021-09-28</th>
      <td>105.290001</td>
      <td>106.750000</td>
      <td>104.730003</td>
      <td>105.730003</td>
      <td>105.637512</td>
    </tr>
    <tr>
      <th>2021-09-29</th>
      <td>106.000000</td>
      <td>107.000000</td>
      <td>105.309998</td>
      <td>106.279999</td>
      <td>106.187027</td>
    </tr>
  </tbody>
</table>
</div>



## 进行训练、测试集拆分


```python
test_split=round(len(df)*0.20)
test_split
```




    1041




```python
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
df_for_training.shape
df_for_testing.shape
```




    (4162, 5)






    (1041, 5)



## 数据归一化
* MinMaxScaler缩放数据


```python
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
```


```python
df_for_training_scaled
```




    array([[0.85398707, 0.86281807, 0.85292546, 0.8403402 , 0.82180889],
           [0.85533406, 0.85473269, 0.82623316, 0.8122593 , 0.79289309],
           [0.82155169, 0.84260459, 0.81422168, 0.80701755, 0.78749611],
           ...,
           [0.40689652, 0.40362224, 0.41960282, 0.40436458, 0.7632948 ],
           [0.40517242, 0.39995691, 0.41832161, 0.4075738 , 0.76889077],
           [0.40862067, 0.39974127, 0.41426436, 0.39880189, 0.75359571]])




```python
df_for_training_scaled.shape
df_for_testing_scaled.shape
```




    (4162, 5)






    (1041, 5)



## 将数据拆分为X和Y


```python
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)        
```

让我们看看上面的代码中做了什么：

N_past是我们在预测下一个目标值时将在过去查看的步骤数。

这里使用30，意味着将使用过去的30个值(包括目标列在内的所有特性)来预测第31个目标值。

因此，在trainX中我们会有所有的特征值，而在trainY中我们只有目标值。

让我们分解for循环的每一部分

对于训练，dataset = df_for_training_scaled, n_past=30

当i= 30:

data_X.addend (df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])

从n_past开始的范围是30，所以第一次数据范围将是-[30 - 30,30,0:5] 相当于 [0:30,0:5]

因此在dataX列表中，df_for_training_scaled[0:30,0:5]数组将第一次出现。

现在, dataY.append(df_for_training_scaled[i,0])

i = 30，所以它将只取第30行开始的open(因为在预测中，我们只需要open列，所以列范围仅为0，表示open列)。

第一次在dataY列表中存储df_for_training_scaled[30,0]值。

所以包含5列的前30行存储在dataX中，只有open列的第31行存储在dataY中。然后我们将dataX和dataY列表转换为数组，它们以数组格式在LSTM中进行训练。


```python
trainX,trainY=createXY(df_for_training_scaled,30)
trainX.shape
trainY.shape
```




    (4132, 30, 5)






    (4132,)



4132 是 trainX 中可用的数组总数，每个数组共有 30 行和 5 列， 在每个数组的 trainY 中，我们都有下一个目标值来训练模型。


```python
trainX[0]
trainY[0]
```




    array([[0.85398707, 0.86281807, 0.85292546, 0.8403402 , 0.82180889],
           [0.85533406, 0.85473269, 0.82623316, 0.8122593 , 0.79289309],
           [0.82155169, 0.84260459, 0.81422168, 0.80701755, 0.78749611],
           [0.81918098, 0.84303579, 0.82319031, 0.8470261 , 0.8286929 ],
           [0.86206895, 0.85769729, 0.84753366, 0.84124952, 0.8227448 ],
           [0.85668106, 0.85295391, 0.85479397, 0.84659822, 0.82825271],
           [0.85129307, 0.85661925, 0.85372629, 0.84766796, 0.82935384],
           [0.8540948 , 0.88249248, 0.85799703, 0.88125807, 0.86394198],
           [0.88577588, 0.88227684, 0.88255396, 0.87590928, 0.85843431],
           [0.88189657, 0.87602422, 0.87016871, 0.86200256, 0.84411479],
           [0.88362063, 0.88357053, 0.87892376, 0.86606763, 0.84829995],
           [0.87047413, 0.86200956, 0.84390347, 0.8344031 , 0.81569511],
           [0.83857758, 0.87645542, 0.84966903, 0.87398378, 0.85645156],
           [0.88168102, 0.88012075, 0.88105913, 0.86649551, 0.84874037],
           [0.87090514, 0.86287196, 0.85949177, 0.84724008, 0.82891342],
           [0.85237065, 0.88249248, 0.8601324 , 0.88403941, 0.86680588],
           [0.85668106, 0.86589053, 0.86248134, 0.8630723 , 0.84521599],
           [0.87176726, 0.8870203 , 0.88191333, 0.87783486, 0.86041721],
           [0.88254305, 0.89003888, 0.88298101, 0.86949083, 0.85182511],
           [0.875     , 0.86955587, 0.85821051, 0.86521178, 0.84741896],
           [0.85775864, 0.85877534, 0.83600253, 0.84552848, 0.82715127],
           [0.86745685, 0.88055195, 0.86120008, 0.88403941, 0.86680588],
           [0.87780172, 0.88033639, 0.87828313, 0.88446729, 0.86724607],
           [0.88900862, 0.88551106, 0.84838778, 0.85237489, 0.83420073],
           [0.83512929, 0.83872361, 0.83365367, 0.83975181, 0.82120263],
           [0.83189656, 0.82988361, 0.82532568, 0.81108261, 0.79168158],
           [0.81896551, 0.82341526, 0.82703399, 0.82199401, 0.80649534],
           [0.85129307, 0.85015098, 0.84240872, 0.8292683 , 0.81401255],
           [0.83448273, 0.84282023, 0.84561178, 0.84124952, 0.82639405],
           [0.84913791, 0.84497632, 0.83557547, 0.83889605, 0.82396218]])






    0.8297413960482309




```python
trainX[1]
trainY[1]
```




    array([[0.85533406, 0.85473269, 0.82623316, 0.8122593 , 0.79289309],
           [0.82155169, 0.84260459, 0.81422168, 0.80701755, 0.78749611],
           [0.81918098, 0.84303579, 0.82319031, 0.8470261 , 0.8286929 ],
           [0.86206895, 0.85769729, 0.84753366, 0.84124952, 0.8227448 ],
           [0.85668106, 0.85295391, 0.85479397, 0.84659822, 0.82825271],
           [0.85129307, 0.85661925, 0.85372629, 0.84766796, 0.82935384],
           [0.8540948 , 0.88249248, 0.85799703, 0.88125807, 0.86394198],
           [0.88577588, 0.88227684, 0.88255396, 0.87590928, 0.85843431],
           [0.88189657, 0.87602422, 0.87016871, 0.86200256, 0.84411479],
           [0.88362063, 0.88357053, 0.87892376, 0.86606763, 0.84829995],
           [0.87047413, 0.86200956, 0.84390347, 0.8344031 , 0.81569511],
           [0.83857758, 0.87645542, 0.84966903, 0.87398378, 0.85645156],
           [0.88168102, 0.88012075, 0.88105913, 0.86649551, 0.84874037],
           [0.87090514, 0.86287196, 0.85949177, 0.84724008, 0.82891342],
           [0.85237065, 0.88249248, 0.8601324 , 0.88403941, 0.86680588],
           [0.85668106, 0.86589053, 0.86248134, 0.8630723 , 0.84521599],
           [0.87176726, 0.8870203 , 0.88191333, 0.87783486, 0.86041721],
           [0.88254305, 0.89003888, 0.88298101, 0.86949083, 0.85182511],
           [0.875     , 0.86955587, 0.85821051, 0.86521178, 0.84741896],
           [0.85775864, 0.85877534, 0.83600253, 0.84552848, 0.82715127],
           [0.86745685, 0.88055195, 0.86120008, 0.88403941, 0.86680588],
           [0.87780172, 0.88033639, 0.87828313, 0.88446729, 0.86724607],
           [0.88900862, 0.88551106, 0.84838778, 0.85237489, 0.83420073],
           [0.83512929, 0.83872361, 0.83365367, 0.83975181, 0.82120263],
           [0.83189656, 0.82988361, 0.82532568, 0.81108261, 0.79168158],
           [0.81896551, 0.82341526, 0.82703399, 0.82199401, 0.80649534],
           [0.85129307, 0.85015098, 0.84240872, 0.8292683 , 0.81401255],
           [0.83448273, 0.84282023, 0.84561178, 0.84124952, 0.82639405],
           [0.84913791, 0.84497632, 0.83557547, 0.83889605, 0.82396218],
           [0.8297414 , 0.8236309 , 0.80247706, 0.79482243, 0.77841557]])






    0.7812499709321126



如果查看 trainX[1] 值，会发现到它与 trainX[0] 中的数据相同（第一行除外），因为我们将看到前 30 行来预测第 31 行，在第一次预测之后它会自动移动 到第 2 行并取下一个 30 值来预测下一个目标值。

让我们用一种简单的格式来解释这一切——

trainX — — →trainY  
[0 : 30,0:5] → [30,0]  
[1:31, 0:5] → [31,0]  
[2:32,0:5] →[32,0]  


```python
testX,testY=createXY(df_for_testing_scaled,30)
testX.shape
testY.shape
```




    (1011, 30, 5)






    (1011,)



## 训练模型
* 使用 girdsearchCV 进行一些超参数调整以找到基础模型。

### GridSearchCV
* GridSearchCV 是 scikit-learn 中用于超参数优化的强大工具，它通过网格搜索和交叉验证来寻找模型的最佳参数组合。
* 核心功能
    * 网格搜索：
        * 自动遍历用户定义的所有参数组合
        * 穷举式搜索给定的参数空间
    * 交叉验证：
        * 对每种参数组合进行K折交叉验证
        * 评估参数组合的泛化性能
* 主要参数
```python
GridSearchCV(
    estimator,          # 要调参的模型/估计器
    param_grid,         # 参数网格(字典或字典列表)
    scoring=None,       # 评估指标
    n_jobs=None,       # 并行作业数(-1使用所有CPU)
    refit=True,        # 是否用最佳参数重新拟合
    cv=None,           # 交叉验证策略
    verbose=0,         # 详细程度
    pre_dispatch='2*n_jobs',
    error_score=np.nan,
    return_train_score=False
)
```
* 工作原理
    * 根据 param_grid 生成所有可能的参数组合
    * 对每种组合：
        * 使用交叉验证训练模型
        * 计算验证集上的平均得分
    * 选择得分最高的参数组合作为最佳参数
    * (如果 refit=True) 用最佳参数在整个训练集上重新训练模型
* 输出结果分析
    * 最佳模型信息：
        * best_estimator_: 最佳参数对应的模型
        * best_params_: 最佳参数组合
        * best_score_: 最佳模型的平均交叉验证得分
    * 完整搜索记录：
        * cv_results_: 包含所有参数组合的详细测试结果
        * 可转换为DataFrame进行更方便的分析
    * 优点
        * 自动化：自动完成繁琐的参数搜索过程
        * 可靠性：通过交叉验证评估参数性能
        * 灵活性：支持自定义评分函数和交叉验证策略
        * 可重复性：记录所有尝试的参数组合和结果
    * 注意事项
        * 参数网格过大会导致计算成本高
        * 对于连续参数，可能需要先大范围粗搜，再小范围精搜
        * 可以使用 RandomizedSearchCV 替代，当参数空间很大时更高效
* GridSearchCV 是机器学习工作流中参数调优的重要工具，能显著提升模型性能。
* [sklearn中的GridSearchCV方法详解](https://www.cnblogs.com/dalege/p/14175192.html)


```python
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
```


```python
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1)

parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
```

如果你想为你的模型做更多的超参数调整，也可以添加更多的层。但是如果数据集非常大建议增加 LSTM 模型中的时期和单位。

在第一个 LSTM 层中看到输入形状为 (30,5)。它来自 trainX 形状。(trainX.shape[1],trainX.shape[2]) → (30,5)

现在让我们将模型拟合到 trainX 和 trainY 数据中。


```python
grid_search = grid_search.fit(trainX,trainY, validation_data=(testX,testY))
```

    Epoch 1/8
    130/130 [==============================] - 6s 49ms/step - loss: 0.0036 - val_loss: 0.0013
    Epoch 2/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0012 - val_loss: 9.0601e-04
    Epoch 3/8
    130/130 [==============================] - 5s 37ms/step - loss: 0.0010 - val_loss: 0.0011
    Epoch 4/8
    130/130 [==============================] - 5s 37ms/step - loss: 9.0804e-04 - val_loss: 8.0079e-04
    Epoch 5/8
    130/130 [==============================] - 5s 38ms/step - loss: 8.5942e-04 - val_loss: 9.9553e-04
    Epoch 6/8
    130/130 [==============================] - 5s 38ms/step - loss: 7.6868e-04 - val_loss: 9.4733e-04
    Epoch 7/8
    130/130 [==============================] - 5s 37ms/step - loss: 7.0164e-04 - val_loss: 9.0523e-04
    Epoch 8/8
    130/130 [==============================] - 5s 37ms/step - loss: 6.6010e-04 - val_loss: 4.5182e-04
    130/130 [==============================] - 1s 8ms/step - loss: 0.0089
    Epoch 1/8
    130/130 [==============================] - 7s 54ms/step - loss: 0.0085 - val_loss: 5.9935e-04
    Epoch 2/8
    130/130 [==============================] - 5s 41ms/step - loss: 0.0031 - val_loss: 4.6387e-04
    Epoch 3/8
    130/130 [==============================] - 5s 38ms/step - loss: 0.0027 - val_loss: 4.1539e-04
    Epoch 4/8
    130/130 [==============================] - 5s 38ms/step - loss: 0.0026 - val_loss: 1.9821e-04
    Epoch 5/8
    130/130 [==============================] - 5s 37ms/step - loss: 0.0023 - val_loss: 3.6139e-04
    Epoch 6/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0022 - val_loss: 1.7607e-04
    Epoch 7/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0022 - val_loss: 1.3555e-04
    Epoch 8/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0019 - val_loss: 1.9315e-04
    130/130 [==============================] - 1s 8ms/step - loss: 4.6882e-04
    Epoch 1/8
    130/130 [==============================] - 7s 51ms/step - loss: 0.0940 - val_loss: 0.0092
    Epoch 2/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0844 - val_loss: 0.0079
    Epoch 3/8
    130/130 [==============================] - 4s 33ms/step - loss: 0.0760 - val_loss: 0.0066
    Epoch 4/8
    130/130 [==============================] - 5s 35ms/step - loss: 0.0668 - val_loss: 0.0053
    Epoch 5/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.0598 - val_loss: 0.0042
    Epoch 6/8
    130/130 [==============================] - 4s 34ms/step - loss: 0.0500 - val_loss: 0.0032
    Epoch 7/8
    130/130 [==============================] - 4s 34ms/step - loss: 0.0426 - val_loss: 0.0024
    Epoch 8/8
    130/130 [==============================] - 5s 35ms/step - loss: 0.0351 - val_loss: 0.0018
    130/130 [==============================] - 1s 8ms/step - loss: 0.1221
    Epoch 1/8
    130/130 [==============================] - 6s 48ms/step - loss: 0.3187 - val_loss: 0.0125
    Epoch 2/8
    130/130 [==============================] - 5s 40ms/step - loss: 0.2907 - val_loss: 0.0105
    Epoch 3/8
    130/130 [==============================] - 5s 37ms/step - loss: 0.2625 - val_loss: 0.0086
    Epoch 4/8
    130/130 [==============================] - 5s 36ms/step - loss: 0.2357 - val_loss: 0.0068
    Epoch 5/8
    130/130 [==============================] - 5s 35ms/step - loss: 0.2104 - val_loss: 0.0052
    Epoch 6/8
    130/130 [==============================] - 4s 33ms/step - loss: 0.1771 - val_loss: 0.0037
    Epoch 7/8
    130/130 [==============================] - 4s 33ms/step - loss: 0.1528 - val_loss: 0.0026
    Epoch 8/8
    130/130 [==============================] - 4s 33ms/step - loss: 0.1247 - val_loss: 0.0017
    130/130 [==============================] - 1s 8ms/step - loss: 0.0297
    Epoch 1/10
    130/130 [==============================] - 7s 56ms/step - loss: 0.0065 - val_loss: 0.0039
    Epoch 2/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.0013 - val_loss: 0.0021
    Epoch 3/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.0011 - val_loss: 0.0026
    Epoch 4/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.0011 - val_loss: 0.0018
    Epoch 5/10
    130/130 [==============================] - 5s 37ms/step - loss: 9.1434e-04 - val_loss: 0.0021
    Epoch 6/10
    130/130 [==============================] - 5s 36ms/step - loss: 8.1586e-04 - val_loss: 0.0023
    Epoch 7/10
    130/130 [==============================] - 5s 35ms/step - loss: 7.6828e-04 - val_loss: 0.0011
    Epoch 8/10
    130/130 [==============================] - 5s 36ms/step - loss: 8.3430e-04 - val_loss: 0.0025
    Epoch 9/10
    130/130 [==============================] - 5s 35ms/step - loss: 7.0253e-04 - val_loss: 0.0015
    Epoch 10/10
    130/130 [==============================] - 5s 36ms/step - loss: 6.6141e-04 - val_loss: 0.0017
    130/130 [==============================] - 1s 7ms/step - loss: 0.0075
    Epoch 1/10
    130/130 [==============================] - 7s 52ms/step - loss: 0.0142 - val_loss: 4.6565e-04
    Epoch 2/10
    130/130 [==============================] - 5s 39ms/step - loss: 0.0029 - val_loss: 3.9623e-04
    Epoch 3/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.0027 - val_loss: 3.1237e-04
    Epoch 4/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.0026 - val_loss: 1.8539e-04
    Epoch 5/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0026 - val_loss: 1.6375e-04
    Epoch 6/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.0024 - val_loss: 1.5019e-04
    Epoch 7/10
    130/130 [==============================] - 4s 35ms/step - loss: 0.0021 - val_loss: 1.4495e-04
    Epoch 8/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0020 - val_loss: 1.3515e-04
    Epoch 9/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0020 - val_loss: 1.3896e-04
    Epoch 10/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0018 - val_loss: 1.6730e-04
    130/130 [==============================] - 1s 7ms/step - loss: 1.3803e-04
    Epoch 1/10
    130/130 [==============================] - 7s 52ms/step - loss: 0.0274 - val_loss: 0.0015
    Epoch 2/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.0223 - val_loss: 0.0011
    Epoch 3/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.0188 - val_loss: 7.6450e-04
    Epoch 4/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0148 - val_loss: 5.9292e-04
    Epoch 5/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0120 - val_loss: 5.6491e-04
    Epoch 6/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0093 - val_loss: 6.6723e-04
    Epoch 7/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0069 - val_loss: 8.7330e-04
    Epoch 8/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0056 - val_loss: 0.0012
    Epoch 9/10
    130/130 [==============================] - 4s 33ms/step - loss: 0.0047 - val_loss: 0.0015
    Epoch 10/10
    130/130 [==============================] - 4s 32ms/step - loss: 0.0039 - val_loss: 0.0018
    130/130 [==============================] - 1s 8ms/step - loss: 0.0226
    Epoch 1/10
    130/130 [==============================] - 7s 50ms/step - loss: 0.1815 - val_loss: 0.0096
    Epoch 2/10
    130/130 [==============================] - 5s 37ms/step - loss: 0.1596 - val_loss: 0.0075
    Epoch 3/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.1352 - val_loss: 0.0056
    Epoch 4/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.1143 - val_loss: 0.0040
    Epoch 5/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0912 - val_loss: 0.0027
    Epoch 6/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0726 - val_loss: 0.0017
    Epoch 7/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0580 - val_loss: 0.0010
    Epoch 8/10
    130/130 [==============================] - 5s 35ms/step - loss: 0.0449 - val_loss: 7.2442e-04
    Epoch 9/10
    130/130 [==============================] - 5s 36ms/step - loss: 0.0343 - val_loss: 7.3390e-04
    Epoch 10/10
    130/130 [==============================] - 4s 34ms/step - loss: 0.0243 - val_loss: 0.0010
    130/130 [==============================] - 1s 8ms/step - loss: 0.0031
    Epoch 1/8
    104/104 [==============================] - 6s 53ms/step - loss: 0.0039 - val_loss: 0.0011
    Epoch 2/8
    104/104 [==============================] - 4s 40ms/step - loss: 0.0011 - val_loss: 0.0015
    Epoch 3/8
    104/104 [==============================] - 4s 41ms/step - loss: 0.0010 - val_loss: 0.0011
    Epoch 4/8
    104/104 [==============================] - 4s 40ms/step - loss: 8.5237e-04 - val_loss: 9.7897e-04
    Epoch 5/8
    104/104 [==============================] - 4s 39ms/step - loss: 8.5551e-04 - val_loss: 8.7465e-04
    Epoch 6/8
    104/104 [==============================] - 4s 40ms/step - loss: 7.7916e-04 - val_loss: 0.0012
    Epoch 7/8
    104/104 [==============================] - 4s 38ms/step - loss: 7.5600e-04 - val_loss: 9.9559e-04
    Epoch 8/8
    104/104 [==============================] - 4s 39ms/step - loss: 8.1150e-04 - val_loss: 8.5240e-04
    104/104 [==============================] - 1s 8ms/step - loss: 0.0136
    Epoch 1/8
    104/104 [==============================] - 6s 60ms/step - loss: 0.0126 - val_loss: 0.0015
    Epoch 2/8
    104/104 [==============================] - 4s 38ms/step - loss: 0.0031 - val_loss: 9.3226e-04
    Epoch 3/8
    104/104 [==============================] - 4s 37ms/step - loss: 0.0026 - val_loss: 1.9322e-04
    Epoch 4/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0025 - val_loss: 2.0439e-04
    Epoch 5/8
    104/104 [==============================] - 4s 35ms/step - loss: 0.0024 - val_loss: 3.4317e-04
    Epoch 6/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0024 - val_loss: 3.2410e-04
    Epoch 7/8
    104/104 [==============================] - 4s 35ms/step - loss: 0.0023 - val_loss: 2.8274e-04
    Epoch 8/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0021 - val_loss: 2.9096e-04
    104/104 [==============================] - 1s 8ms/step - loss: 1.8789e-04
    Epoch 1/8
    104/104 [==============================] - 6s 53ms/step - loss: 0.1212 - val_loss: 0.0151
    Epoch 2/8
    104/104 [==============================] - 4s 38ms/step - loss: 0.1124 - val_loss: 0.0131
    Epoch 3/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.1003 - val_loss: 0.0113
    Epoch 4/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0905 - val_loss: 0.0096
    Epoch 5/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0812 - val_loss: 0.0080
    Epoch 6/8
    104/104 [==============================] - 4s 35ms/step - loss: 0.0715 - val_loss: 0.0066
    Epoch 7/8
    104/104 [==============================] - 4s 37ms/step - loss: 0.0623 - val_loss: 0.0053
    Epoch 8/8
    104/104 [==============================] - 4s 35ms/step - loss: 0.0537 - val_loss: 0.0041
    104/104 [==============================] - 1s 7ms/step - loss: 0.1303
    Epoch 1/8
    104/104 [==============================] - 6s 54ms/step - loss: 0.1396 - val_loss: 0.0048
    Epoch 2/8
    104/104 [==============================] - 4s 40ms/step - loss: 0.1265 - val_loss: 0.0038
    Epoch 3/8
    104/104 [==============================] - 4s 38ms/step - loss: 0.1096 - val_loss: 0.0030
    Epoch 4/8
    104/104 [==============================] - 4s 38ms/step - loss: 0.0969 - val_loss: 0.0023
    Epoch 5/8
    104/104 [==============================] - 4s 39ms/step - loss: 0.0825 - val_loss: 0.0016
    Epoch 6/8
    104/104 [==============================] - 4s 37ms/step - loss: 0.0709 - val_loss: 0.0012
    Epoch 7/8
    104/104 [==============================] - 4s 36ms/step - loss: 0.0593 - val_loss: 7.9644e-04
    Epoch 8/8
    104/104 [==============================] - 4s 35ms/step - loss: 0.0494 - val_loss: 5.8013e-04
    104/104 [==============================] - 1s 8ms/step - loss: 0.0083
    Epoch 1/10
    104/104 [==============================] - 5s 51ms/step - loss: 0.0036 - val_loss: 0.0028
    Epoch 2/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.0011 - val_loss: 0.0023
    Epoch 3/10
    104/104 [==============================] - 4s 39ms/step - loss: 9.7720e-04 - val_loss: 0.0015
    Epoch 4/10
    104/104 [==============================] - 4s 40ms/step - loss: 9.9515e-04 - val_loss: 0.0014
    Epoch 5/10
    104/104 [==============================] - 4s 38ms/step - loss: 9.2495e-04 - val_loss: 7.0709e-04
    Epoch 6/10
    104/104 [==============================] - 4s 37ms/step - loss: 8.0184e-04 - val_loss: 0.0010
    Epoch 7/10
    104/104 [==============================] - 4s 43ms/step - loss: 7.8312e-04 - val_loss: 9.5966e-04
    Epoch 8/10
    104/104 [==============================] - 4s 38ms/step - loss: 7.9821e-04 - val_loss: 8.2732e-04
    Epoch 9/10
    104/104 [==============================] - 4s 38ms/step - loss: 6.6008e-04 - val_loss: 5.0335e-04
    Epoch 10/10
    104/104 [==============================] - 4s 39ms/step - loss: 6.7522e-04 - val_loss: 5.9923e-04
    104/104 [==============================] - 1s 9ms/step - loss: 0.0062
    Epoch 1/10
    104/104 [==============================] - 6s 59ms/step - loss: 0.0089 - val_loss: 5.6323e-04
    Epoch 2/10
    104/104 [==============================] - 4s 42ms/step - loss: 0.0030 - val_loss: 4.1378e-04
    Epoch 3/10
    104/104 [==============================] - 4s 41ms/step - loss: 0.0029 - val_loss: 7.4584e-04
    Epoch 4/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.0029 - val_loss: 7.1552e-04
    Epoch 5/10
    104/104 [==============================] - 4s 38ms/step - loss: 0.0023 - val_loss: 4.5248e-04
    Epoch 6/10
    104/104 [==============================] - 4s 36ms/step - loss: 0.0025 - val_loss: 2.7011e-04
    Epoch 7/10
    104/104 [==============================] - 4s 37ms/step - loss: 0.0022 - val_loss: 1.7423e-04
    Epoch 8/10
    104/104 [==============================] - 4s 37ms/step - loss: 0.0021 - val_loss: 1.7378e-04
    Epoch 9/10
    104/104 [==============================] - 4s 37ms/step - loss: 0.0018 - val_loss: 1.1696e-04
    Epoch 10/10
    104/104 [==============================] - 4s 38ms/step - loss: 0.0019 - val_loss: 1.1212e-04
    104/104 [==============================] - 1s 9ms/step - loss: 1.4757e-04
    Epoch 1/10
    104/104 [==============================] - 5s 48ms/step - loss: 0.1241 - val_loss: 0.0161
    Epoch 2/10
    104/104 [==============================] - 4s 34ms/step - loss: 0.1151 - val_loss: 0.0143
    Epoch 3/10
    104/104 [==============================] - 4s 36ms/step - loss: 0.1045 - val_loss: 0.0125
    Epoch 4/10
    104/104 [==============================] - 4s 38ms/step - loss: 0.0945 - val_loss: 0.0107
    Epoch 5/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.0835 - val_loss: 0.0090
    Epoch 6/10
    104/104 [==============================] - 4s 35ms/step - loss: 0.0740 - val_loss: 0.0074
    Epoch 7/10
    104/104 [==============================] - 4s 36ms/step - loss: 0.0641 - val_loss: 0.0060
    Epoch 8/10
    104/104 [==============================] - 4s 36ms/step - loss: 0.0547 - val_loss: 0.0047
    Epoch 9/10
    104/104 [==============================] - 4s 35ms/step - loss: 0.0472 - val_loss: 0.0036
    Epoch 10/10
    104/104 [==============================] - 4s 35ms/step - loss: 0.0399 - val_loss: 0.0026
    104/104 [==============================] - 1s 9ms/step - loss: 0.1107
    Epoch 1/10
    104/104 [==============================] - 6s 54ms/step - loss: 0.3344 - val_loss: 0.0168
    Epoch 2/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.3071 - val_loss: 0.0145
    Epoch 3/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.2819 - val_loss: 0.0123
    Epoch 4/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.2576 - val_loss: 0.0102
    Epoch 5/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.2314 - val_loss: 0.0083
    Epoch 6/10
    104/104 [==============================] - 4s 40ms/step - loss: 0.2071 - val_loss: 0.0065
    Epoch 7/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.1818 - val_loss: 0.0050
    Epoch 8/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.1585 - val_loss: 0.0036
    Epoch 9/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.1352 - val_loss: 0.0026
    Epoch 10/10
    104/104 [==============================] - 4s 39ms/step - loss: 0.1159 - val_loss: 0.0018
    104/104 [==============================] - 1s 9ms/step - loss: 0.0287
    Epoch 1/10
    207/207 [==============================] - 10s 47ms/step - loss: 0.0078 - val_loss: 1.7512e-04
    Epoch 2/10
    207/207 [==============================] - 8s 38ms/step - loss: 0.0021 - val_loss: 2.4985e-04
    Epoch 3/10
    207/207 [==============================] - 8s 37ms/step - loss: 0.0018 - val_loss: 1.4491e-04
    Epoch 4/10
    207/207 [==============================] - 7s 36ms/step - loss: 0.0016 - val_loss: 1.4965e-04
    Epoch 5/10
    207/207 [==============================] - 7s 36ms/step - loss: 0.0014 - val_loss: 1.5778e-04
    Epoch 6/10
    207/207 [==============================] - 7s 36ms/step - loss: 0.0013 - val_loss: 1.8984e-04
    Epoch 7/10
    207/207 [==============================] - 7s 36ms/step - loss: 0.0012 - val_loss: 1.2000e-04
    Epoch 8/10
    207/207 [==============================] - 7s 35ms/step - loss: 0.0011 - val_loss: 1.4246e-04
    Epoch 9/10
    207/207 [==============================] - 7s 36ms/step - loss: 9.6784e-04 - val_loss: 1.0352e-04
    Epoch 10/10
    207/207 [==============================] - 7s 36ms/step - loss: 9.4916e-04 - val_loss: 1.0256e-04
    

### 检查模型的最佳参数


```python
grid_search.best_params_
```




    {'batch_size': 20, 'epochs': 10, 'optimizer': 'adam'}



* 将最佳模型保存在 my_model 变量中。


```python
my_model=grid_search.best_estimator_.model
```


```python
my_model
```




    <tensorflow.python.keras.engine.sequential.Sequential at 0x243676baa30>



* 其他查询


```python
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)
```

    最佳参数: {'batch_size': 20, 'epochs': 10, 'optimizer': 'adam'}
    最佳得分: -0.003174197052430827
    

* 完整的交叉验证结果
* cv_results_ 包含所有参数组合的详细测试结果，典型的 cv_results_ DataFrame 包含以下列：
    * mean_fit_time: 平均拟合时间
    * mean_score_time: 平均评分时间
    * mean_test_score: 平均测试得分
    * param_[参数名]: 各种参数的取值
    * params: 完整的参数组合
    * rank_test_score: 按测试得分排序的排名
    * split0_test_score, split1_test_score, ...: 每个交叉验证折的得分


```python
results = pd.DataFrame(grid_search.cv_results_)
results
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_batch_size</th>
      <th>param_epochs</th>
      <th>param_optimizer</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46.414048</td>
      <td>0.351675</td>
      <td>1.109864</td>
      <td>0.009797</td>
      <td>16</td>
      <td>8</td>
      <td>adam</td>
      <td>{'batch_size': 16, 'epochs': 8, 'optimizer': '...</td>
      <td>-0.008890</td>
      <td>-0.000469</td>
      <td>-0.004679</td>
      <td>0.004210</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43.543633</td>
      <td>0.053244</td>
      <td>1.117882</td>
      <td>0.019736</td>
      <td>16</td>
      <td>8</td>
      <td>Adadelta</td>
      <td>{'batch_size': 16, 'epochs': 8, 'optimizer': '...</td>
      <td>-0.122149</td>
      <td>-0.029667</td>
      <td>-0.075908</td>
      <td>0.046241</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.451011</td>
      <td>0.261672</td>
      <td>1.030178</td>
      <td>0.018657</td>
      <td>16</td>
      <td>10</td>
      <td>adam</td>
      <td>{'batch_size': 16, 'epochs': 10, 'optimizer': ...</td>
      <td>-0.007537</td>
      <td>-0.000138</td>
      <td>-0.003837</td>
      <td>0.003699</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52.616302</td>
      <td>0.536564</td>
      <td>1.084641</td>
      <td>0.008760</td>
      <td>16</td>
      <td>10</td>
      <td>Adadelta</td>
      <td>{'batch_size': 16, 'epochs': 10, 'optimizer': ...</td>
      <td>-0.022645</td>
      <td>-0.003101</td>
      <td>-0.012873</td>
      <td>0.009772</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38.676844</td>
      <td>0.791599</td>
      <td>0.922409</td>
      <td>0.014502</td>
      <td>20</td>
      <td>8</td>
      <td>adam</td>
      <td>{'batch_size': 20, 'epochs': 8, 'optimizer': '...</td>
      <td>-0.013624</td>
      <td>-0.000188</td>
      <td>-0.006906</td>
      <td>0.006718</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>37.839067</td>
      <td>0.422257</td>
      <td>0.889523</td>
      <td>0.045836</td>
      <td>20</td>
      <td>8</td>
      <td>Adadelta</td>
      <td>{'batch_size': 20, 'epochs': 8, 'optimizer': '...</td>
      <td>-0.130294</td>
      <td>-0.008277</td>
      <td>-0.069285</td>
      <td>0.061008</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>47.657228</td>
      <td>0.018644</td>
      <td>1.027748</td>
      <td>0.023700</td>
      <td>20</td>
      <td>10</td>
      <td>adam</td>
      <td>{'batch_size': 20, 'epochs': 10, 'optimizer': ...</td>
      <td>-0.006201</td>
      <td>-0.000148</td>
      <td>-0.003174</td>
      <td>0.003027</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46.334166</td>
      <td>2.149322</td>
      <td>0.980408</td>
      <td>0.005555</td>
      <td>20</td>
      <td>10</td>
      <td>Adadelta</td>
      <td>{'batch_size': 20, 'epochs': 10, 'optimizer': ...</td>
      <td>-0.110695</td>
      <td>-0.028698</td>
      <td>-0.069696</td>
      <td>0.040999</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



* 可视化结果


```python
import matplotlib.pyplot as plt

results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['rank_test_score'], results['mean_test_score'], 'o-')
plt.xlabel('C value')
plt.ylabel('Mean Test Score')
plt.xscale('log')  # 如果C是对数分布的
plt.show()
```




    <Figure size 1000x600 with 0 Axes>






    [<matplotlib.lines.Line2D at 0x2437575fe50>]






    Text(0.5, 0, 'C value')






    Text(0, 0.5, 'Mean Test Score')




    
![png](output_42_4.png)
    


### 用测试数据集测试模型


```python
prediction=my_model.predict(testX)
```


```python
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)
print("testY\n", testY)
print("\ntestYShape-",testY.shape)
```

    prediction
     [[0.3667599 ]
     [0.3702871 ]
     [0.3733918 ]
     ...
     [0.13179763]
     [0.13430063]
     [0.13725263]]
    
    Prediction Shape- (1011, 1)
    testY
     [0.39159482 0.3965517  0.39331896 ... 0.14744613 0.1495194  0.15150862]
    
    testYShape- (1011,)
    

testY 和 prediction 的长度是一样的。现在可以将 testY 与预测进行比较。

但是我们一开始就对数据进行了缩放，所以首先我们必须做一些逆缩放过程。
因为在缩放数据时，我们每行有 5 列，现在我们只有 1 列是目标列。

所以我们必须改变形状来使用 inverse_transform


```python
prediction_copies_array = np.repeat(prediction,5, axis=-1)
```

5 列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。


```python
prediction_copies_array.shape
```




    (1011, 5)




```python
prediction_copies_array
```




    array([[0.3667599 , 0.3667599 , 0.3667599 , 0.3667599 , 0.3667599 ],
           [0.3702871 , 0.3702871 , 0.3702871 , 0.3702871 , 0.3702871 ],
           [0.3733918 , 0.3733918 , 0.3733918 , 0.3733918 , 0.3733918 ],
           ...,
           [0.13179763, 0.13179763, 0.13179763, 0.13179763, 0.13179763],
           [0.13430063, 0.13430063, 0.13430063, 0.13430063, 0.13430063],
           [0.13725263, 0.13725263, 0.13725263, 0.13725263, 0.13725263]],
          dtype=float32)




```python
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]
```

但是逆变换后的第一列是我们需要的，所以我们在最后使用了 → [:,0]。

现在将这个 pred 值与 testY 进行比较，但是 testY 也是按比例缩放的，也需要使用与上述相同的代码进行逆变换。


```python
pred
```




    array([182.82816 , 184.08708 , 185.19524 , ...,  98.96468 ,  99.85807 ,
           100.911705], dtype=float32)




```python
original_copies_array = np.repeat(testY,5, axis=-1)

original_copies_array.shape

original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]
```




    (5055,)




```python
print("Pred Values-- " ,pred)
print("\nOriginal Values-- ",original)
```

    Pred Values--  [182.82816  184.08708  185.19524  ...  98.96468   99.85807  100.911705]
    
    Original Values--  [191.692307 193.461533 192.307693 ... 104.550003 105.290001 106.      ]
    

### 最后绘制一个图来对比我们的 pred 和原始数据。


```python
import matplotlib.pyplot as plt
```


```python
plt.plot(original, color = 'red', label = 'Real  Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()
```




    [<matplotlib.lines.Line2D at 0x24375c5e850>]






    [<matplotlib.lines.Line2D at 0x24375c5ebe0>]






    Text(0.5, 1.0, ' Stock Price Prediction')






    Text(0.5, 0, 'Time')






    Text(0, 0.5, ' Stock Price')






    <matplotlib.legend.Legend at 0x24375c5e880>




    
![png](output_58_6.png)
    


## 预测未来值
* 从主 df 数据集中获取我们在开始时加载的最后 30 个值[为什么是 30？因为这是我们想要的过去值的数量，来预测第 31 个值]


```python
df_30_days_past=df.iloc[-30:,:]
df_30_days_past.tail()
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-09-23</th>
      <td>99.529999</td>
      <td>104.080002</td>
      <td>99.519997</td>
      <td>102.959999</td>
      <td>102.789993</td>
    </tr>
    <tr>
      <th>2021-09-24</th>
      <td>102.660004</td>
      <td>104.199997</td>
      <td>102.599998</td>
      <td>103.800003</td>
      <td>103.709198</td>
    </tr>
    <tr>
      <th>2021-09-27</th>
      <td>104.550003</td>
      <td>106.330002</td>
      <td>104.389999</td>
      <td>105.349998</td>
      <td>105.257835</td>
    </tr>
    <tr>
      <th>2021-09-28</th>
      <td>105.290001</td>
      <td>106.750000</td>
      <td>104.730003</td>
      <td>105.730003</td>
      <td>105.637512</td>
    </tr>
    <tr>
      <th>2021-09-29</th>
      <td>106.000000</td>
      <td>107.000000</td>
      <td>105.309998</td>
      <td>106.279999</td>
      <td>106.187027</td>
    </tr>
  </tbody>
</table>
</div>



可以看到有包括目标列（“Open”）在内的所有列。现在让我们预测未来的 30 个值。

在多元时间序列预测中，需要通过使用不同的特征来预测单列，所以在进行预测时我们需要使用特征值（目标列除外）来进行即将到来的预测。

这里我们需要“High”、“Low”、“Close”、“Adj Close”列的即将到来的 30 个值来对“Open”列进行预测。


```python
df_30_days_future=pd.read_csv("test.csv",parse_dates=["Date"],index_col=[0])
df_30_days_future.shape
```




    (30, 4)




```python
df_30_days_future
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
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-09-30</th>
      <td>107.089996</td>
      <td>102.949997</td>
      <td>103.029999</td>
      <td>102.939865</td>
    </tr>
    <tr>
      <th>2021-10-01</th>
      <td>106.389999</td>
      <td>103.669998</td>
      <td>105.820000</td>
      <td>105.727425</td>
    </tr>
    <tr>
      <th>2021-10-04</th>
      <td>107.080002</td>
      <td>104.599998</td>
      <td>104.900002</td>
      <td>104.808235</td>
    </tr>
    <tr>
      <th>2021-10-05</th>
      <td>106.000000</td>
      <td>103.750000</td>
      <td>104.900002</td>
      <td>104.808235</td>
    </tr>
    <tr>
      <th>2021-10-06</th>
      <td>104.419998</td>
      <td>102.059998</td>
      <td>104.330002</td>
      <td>104.238731</td>
    </tr>
    <tr>
      <th>2021-10-07</th>
      <td>106.529999</td>
      <td>104.330002</td>
      <td>105.510002</td>
      <td>105.417702</td>
    </tr>
    <tr>
      <th>2021-10-08</th>
      <td>106.220001</td>
      <td>104.660004</td>
      <td>104.720001</td>
      <td>104.628387</td>
    </tr>
    <tr>
      <th>2021-10-11</th>
      <td>105.760002</td>
      <td>103.970001</td>
      <td>104.080002</td>
      <td>103.988953</td>
    </tr>
    <tr>
      <th>2021-10-12</th>
      <td>104.040001</td>
      <td>101.559998</td>
      <td>102.720001</td>
      <td>102.630142</td>
    </tr>
    <tr>
      <th>2021-10-13</th>
      <td>103.199997</td>
      <td>101.180000</td>
      <td>102.360001</td>
      <td>102.270454</td>
    </tr>
    <tr>
      <th>2021-10-14</th>
      <td>103.650002</td>
      <td>102.370003</td>
      <td>102.739998</td>
      <td>102.650116</td>
    </tr>
    <tr>
      <th>2021-10-15</th>
      <td>105.900002</td>
      <td>103.190002</td>
      <td>104.410004</td>
      <td>104.318665</td>
    </tr>
    <tr>
      <th>2021-10-18</th>
      <td>104.570000</td>
      <td>103.040001</td>
      <td>104.120003</td>
      <td>104.028915</td>
    </tr>
    <tr>
      <th>2021-10-19</th>
      <td>104.970001</td>
      <td>103.580002</td>
      <td>104.730003</td>
      <td>104.638382</td>
    </tr>
    <tr>
      <th>2021-10-20</th>
      <td>106.019997</td>
      <td>103.870003</td>
      <td>106.000000</td>
      <td>105.907272</td>
    </tr>
    <tr>
      <th>2021-10-21</th>
      <td>106.389999</td>
      <td>103.010002</td>
      <td>103.150002</td>
      <td>103.059761</td>
    </tr>
    <tr>
      <th>2021-10-22</th>
      <td>104.510002</td>
      <td>102.550003</td>
      <td>104.050003</td>
      <td>103.958977</td>
    </tr>
    <tr>
      <th>2021-10-25</th>
      <td>105.989998</td>
      <td>103.330002</td>
      <td>105.300003</td>
      <td>105.207886</td>
    </tr>
    <tr>
      <th>2021-10-26</th>
      <td>110.970001</td>
      <td>105.220001</td>
      <td>107.440002</td>
      <td>107.346008</td>
    </tr>
    <tr>
      <th>2021-10-27</th>
      <td>108.279999</td>
      <td>103.690002</td>
      <td>103.849998</td>
      <td>103.759148</td>
    </tr>
    <tr>
      <th>2021-10-28</th>
      <td>105.379997</td>
      <td>103.099998</td>
      <td>105.260002</td>
      <td>105.167915</td>
    </tr>
    <tr>
      <th>2021-10-29</th>
      <td>105.239998</td>
      <td>104.120003</td>
      <td>104.870003</td>
      <td>104.778259</td>
    </tr>
    <tr>
      <th>2021-11-01</th>
      <td>106.769997</td>
      <td>105.279999</td>
      <td>106.230003</td>
      <td>106.137070</td>
    </tr>
    <tr>
      <th>2021-11-02</th>
      <td>107.139999</td>
      <td>105.300003</td>
      <td>106.690002</td>
      <td>106.596664</td>
    </tr>
    <tr>
      <th>2021-11-03</th>
      <td>106.339996</td>
      <td>104.820000</td>
      <td>105.970001</td>
      <td>105.877296</td>
    </tr>
    <tr>
      <th>2021-11-04</th>
      <td>106.400002</td>
      <td>104.290001</td>
      <td>105.209999</td>
      <td>105.117958</td>
    </tr>
    <tr>
      <th>2021-11-05</th>
      <td>109.650002</td>
      <td>106.849998</td>
      <td>108.739998</td>
      <td>108.644867</td>
    </tr>
    <tr>
      <th>2021-11-08</th>
      <td>110.309998</td>
      <td>108.320000</td>
      <td>108.419998</td>
      <td>108.325150</td>
    </tr>
    <tr>
      <th>2021-11-09</th>
      <td>116.169998</td>
      <td>110.480003</td>
      <td>111.290001</td>
      <td>111.192642</td>
    </tr>
    <tr>
      <th>2021-11-10</th>
      <td>112.680000</td>
      <td>108.110001</td>
      <td>108.959999</td>
      <td>108.864677</td>
    </tr>
  </tbody>
</table>
</div>



剔除“Open”列后，使用模型进行预测之前还需要做以下的操作：

缩放数据，因为删除了‘Open’列，在缩放它之前，添加一个所有值都为“0”的Open列。

缩放后，将未来数据中的“Open”列值替换为“nan”

现在附加 30 天旧值和 30 天新值（其中最后 30 个“打开”值是 nan）


```python
df_30_days_future["Open"]=0
df_30_days_future=df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler.transform(df_30_days_past)
new_scaled_array=scaler.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)
```


```python
full_df.shape
full_df.tail()
```




    (60, 5)






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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>NaN</td>
      <td>0.143640</td>
      <td>0.164162</td>
      <td>0.150135</td>
      <td>0.349458</td>
    </tr>
    <tr>
      <th>56</th>
      <td>NaN</td>
      <td>0.152749</td>
      <td>0.171268</td>
      <td>0.159953</td>
      <td>0.367565</td>
    </tr>
    <tr>
      <th>57</th>
      <td>NaN</td>
      <td>0.154599</td>
      <td>0.175349</td>
      <td>0.159063</td>
      <td>0.365924</td>
    </tr>
    <tr>
      <th>58</th>
      <td>NaN</td>
      <td>0.171024</td>
      <td>0.181345</td>
      <td>0.167045</td>
      <td>0.380645</td>
    </tr>
    <tr>
      <th>59</th>
      <td>NaN</td>
      <td>0.161242</td>
      <td>0.174766</td>
      <td>0.160565</td>
      <td>0.368694</td>
    </tr>
  </tbody>
</table>
</div>



要进行预测必须再次使用 for 循环，我们在拆分 trainX 和 trainY 中的数据时所做的。但是这次我们只有 X，没有 Y 值


```python
full_df_scaled_array=full_df.values
```


```python
full_df_scaled_array.shape
```




    (60, 5)




```python
all_data=[]
time_step=30
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i,0]=prediction
```


```python
all_data
```




    [array([[0.14023124]], dtype=float32),
     array([[0.1419575]], dtype=float32),
     array([[0.14336602]], dtype=float32),
     array([[0.14437674]], dtype=float32),
     array([[0.14490917]], dtype=float32),
     array([[0.14476171]], dtype=float32),
     array([[0.14482723]], dtype=float32),
     array([[0.14481631]], dtype=float32),
     array([[0.14455676]], dtype=float32),
     array([[0.1436392]], dtype=float32),
     array([[0.14232466]], dtype=float32),
     array([[0.14114404]], dtype=float32),
     array([[0.14074759]], dtype=float32),
     array([[0.14059792]], dtype=float32),
     array([[0.1407732]], dtype=float32),
     array([[0.14144544]], dtype=float32),
     array([[0.14173436]], dtype=float32),
     array([[0.14177807]], dtype=float32),
     array([[0.14211234]], dtype=float32),
     array([[0.14355302]], dtype=float32),
     array([[0.14435941]], dtype=float32),
     array([[0.14472127]], dtype=float32),
     array([[0.14476873]], dtype=float32),
     array([[0.14511383]], dtype=float32),
     array([[0.14567792]], dtype=float32),
     array([[0.14602959]], dtype=float32),
     array([[0.14605506]], dtype=float32),
     array([[0.14704606]], dtype=float32),
     array([[0.14847994]], dtype=float32),
     array([[0.15122235]], dtype=float32)]



对于第一个预测，有之前的 30 个值，当 for 循环第一次运行时它会检查前 30 个值并预测第 31 个“Open”数据。

当第二个 for 循环将尝试运行时，它将跳过第一行并尝试获取下 30 个值 [1:31] 。这里会报错错误因为Open列最后一行是 “nan”，所以需要每次都用预测替换“nan”。

最后还需要对预测进行逆变换→


```python
new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,0]
```


```python
y_pred_future_30_days
```




    array([101.97484 , 102.59098 , 103.09371 , 103.45446 , 103.64451 ,
           103.59187 , 103.61526 , 103.61136 , 103.51872 , 103.19122 ,
           102.72203 , 102.300644, 102.159134, 102.10572 , 102.16828 ,
           102.40822 , 102.511345, 102.52694 , 102.646255, 103.16046 ,
           103.44828 , 103.57744 , 103.594376, 103.71755 , 103.91889 ,
           104.04441 , 104.0535  , 104.40721 , 104.91899 , 105.89783 ],
          dtype=float32)



# END!!!!
