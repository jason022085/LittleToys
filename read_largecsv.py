# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:05:27 2019

@author: USER
"""
import pandas as pd
import numpy as np
#%%

file = open("D:/Google 雲端硬碟/A碩班的血汗/中華郵政大數據競賽/初賽資料/ACCS1.csv")
acc = pd.read_csv(file,sep=',',iterator=True)
loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = acc.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
ACC = pd.concat(chunks, ignore_index=True)
#%%
np.save("C:/Users/USER/Desktop/ACC.npy",arr = ACC)
#%%

file = open("D:/Google 雲端硬碟/A碩班的血汗/中華郵政大數據競賽/初賽資料/ACCS1.csv")
acc = pd.read_csv(file,sep=',',iterator=True)
loop = True
chunkSize = 200000
chunks = []
for i in range(100):
        chunk = acc.get_chunk(chunkSize)
        chunks.append(chunk)
    
ACC = pd.concat(chunks, ignore_index=True)
np.save("C:/Users/USER/Desktop/ACC.npy",arr = ACC)
#%%





















#%%這裡開始是CC檔
file = open("D:/Google 雲端硬碟/A碩班的血汗/中華郵政大數據競賽/初賽資料/CC.csv")
colname = range(28)
CC = pd.read_csv(file, names =  colname,   header = None)
#%%

CC24 = CC[18]
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
for i in range(CC24.shape[0]):
    ind = np.str(CC24[i])
    ind = ind.zfill(8)#取前六位
    c1.append(ind[0])
    c2.append(ind[1])
    c3.append(ind[2])
    c4.append(ind[3])
    c5.append(ind[4])
    c6.append(ind[5])
    
#%%
d = {"國內函件":c1,"國際函件":c2,"國內包裹":c3,"國際包裹":c4,"國內快捷":c5,"國際快捷":c6}
CC_new = pd.DataFrame(d,dtype=np.int8)
CC_new.to_csv('out.csv')
#np.save("CC",arr = CC)
#%%