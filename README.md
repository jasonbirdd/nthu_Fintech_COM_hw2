---
title: 'Fintech COM HW2 詹凱錞'
disqus: hackmd
---

Fintech COM HW2 詹凱錞
===

Some modules are required
please install them use
```
>> pip install -r requirements.txt
```

## Fetch TAIEX data
Run the code which is in fetch_data.py.  
There are two ways to fetch the data.  

* One is crawl the data from https://www.twse.com.tw/zh/ (The main website of Taiwan Stock Exchange) using requests.  
* The more efficient way is using yfinance api to get data from https://tw.stock.yahoo.com/  




## HW2 main
### initial  
The HW2 code is in hw2_main.py and import some function in hw2_models.py  
Before analysing the problems, we should import some modules and read the csv we fetched just.
```python
import pandas as pd
import numpy as np
import math
from hw2_models import triple_barrier, bios_MA, RSI
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

TAIEX_df = pd.read_csv("TAIEX_fetch.csv")
TAIEX_df = TAIEX_df.drop(columns=["Adj Close"])
```
### Question2 : Create triple-barrier labels
```python
# In[2]
ret = triple_barrier(TAIEX_df.Close, 1.04, 0.98, 20) #use funciton triple_barrier in hw2_models
TAIEX_df["TB_label"] = ret.triple_barrier_signal 
```
### Question3A : Create Bias Moving average in 5,10,20,60 days
```python
# In[3A] 
moving_avg_day = [5,10,20,60]
for day in moving_avg_day:
    TAIEX_df[f"MA{day}"] = bios_MA(TAIEX_df.Close, day)#use funciton bios_MA in hw2_models
```
### Question3B : Create RSI 14
```python
# In[3B] 
TAIEX_df["RSI14"] = RSI(TAIEX_df.Close,14)#use funciton RSI in hw2_models
```
### Question3C : Create DIF, MACD signal, MACD histogram & save dataframe to csv
```python
# In[3C]
EMA_12 = TAIEX_df['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = TAIEX_df['Close'].ewm(span=26, adjust=False).mean()
TAIEX_df['DIF'] = EMA_12 - EMA_26
TAIEX_df['MACD_signal'] = TAIEX_df['DIF'].ewm(span=9, adjust=False).mean()
TAIEX_df['MACD_his'] = TAIEX_df['DIF']-TAIEX_df['MACD_signal']


TAIEX_df.to_csv("TAIEX_down.csv")
```
### Question4A : calculate the 2 PCA component & Plot the explained variance ratio and its cumulative sum
```python
# In[4A]
TAIEX_array = np.array(TAIEX_df[["Open", "High", "Low", "Close", "Volume", 
                        "MA5", "MA10", "MA20", "MA60", "RSI14", 
                        "DIF", "MACD_signal", "MACD_his"]])
TAIEX_array = TAIEX_array[59:, :]

pca = PCA().fit(TAIEX_array)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("TAIEX")
plt.xlabel('number of principle components')
plt.ylabel('cumulative explained variance')
pca = PCA(n_components = 2)
pca_feature = pca.fit_transform(TAIEX_array)
plt.show()
```
![](https://i.imgur.com/gfEvPHV.png)

### Question4B : Plot data by class labeled in problem 2
```python
# In[4B]
targets = TAIEX_df.TB_label[59:]
for label in set(targets):
    idx = np.where(np.array(targets) == label)[0]
    plt.scatter(pca_feature[idx, 0], pca_feature[idx, 1], label=label)
plt.legend()
plt.title("TAIEX")
plt.show()
```
![](https://i.imgur.com/oXSnP85.png)
