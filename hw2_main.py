# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:20:20 2021

@author: JasonJhan
"""
import pandas as pd
import numpy as np
import math
from hw2_models import triple_barrier, bios_MA, RSI
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

TAIEX_df = pd.read_csv("TAIEX_fetch.csv")
TAIEX_df = TAIEX_df.drop(columns=["Adj Close"])
# In[2]
ret = triple_barrier(TAIEX_df.Close, 1.04, 0.98, 20) #use funciton triple_barrier in hw2_models
TAIEX_df["TB_label"] = ret.triple_barrier_signal
# In[3A] 
moving_avg_day = [5,10,20,60]
for day in moving_avg_day:
    TAIEX_df[f"MA{day}"] = bios_MA(TAIEX_df.Close, day)
# In[3B] 
TAIEX_df["RSI14"] = RSI(TAIEX_df.Close,14)
# In[3C]
EMA_12 = TAIEX_df['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = TAIEX_df['Close'].ewm(span=26, adjust=False).mean()
TAIEX_df['DIF'] = EMA_12 - EMA_26
TAIEX_df['MACD_signal'] = TAIEX_df['DIF'].ewm(span=9, adjust=False).mean()
TAIEX_df['MACD_his'] = TAIEX_df['DIF']-TAIEX_df['MACD_signal']


TAIEX_df.to_csv("TAIEX_down.csv")

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


# In[4B]
targets = TAIEX_df.TB_label[59:]
for label in set(targets):
    idx = np.where(np.array(targets) == label)[0]
    plt.scatter(pca_feature[idx, 0], pca_feature[idx, 1], label=label)
plt.legend()
plt.title("TAIEX")
plt.show()














