#%%載入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#畫圖用套件 
import keras as kr #神經網路常用套件(易)
from time import time #算時間的套件 
import jieba as jb#斷詞用套件
import pyprind
DT = np.load('D:\Anaconda3\mycode\MLlab\DT_new.npz')#取前300字的DT_new(有加入統計字典的)
X_train = DT["article_train"]
X_test = DT["article_test"]
y_train = DT["board_train"]
y_test = DT["board_test"]
X_train = X_train.reshape((72333,300))
X_test = X_test.reshape((8038,300))
#%%建立LSTM模型所需的語法
from keras.models import Sequential
from keras.layers import LSTM,Activation,Dense,Embedding,GRU,CuDNNLSTM #flatten是要將矩陣拉平成向量
from keras.layers import BatchNormalization,Dropout,LeakyReLU,Bidirectional
from keras.optimizers import SGD,Adam,RMSprop#RMS適合RNN
from keras import regularizers
#%%
En = Sequential()
En.add(Dense(300,input_dim=300))
En.add(Dense(10))
En.summary()
#%%
De = Sequential()
De.add(Dense(10,input_dim=10))
De.add(Dense(300))
De.summary()
#%%
auto = Sequential()
auto.add(En)
auto.add(De)
auto.compile(loss = "kullback_leibler_divergence",optimizer = "Nadam")
#%%
auto.fit(X_train,X_train,epochs=10,batch_size=128)
#%%讀取資料檔
'''
此資料檔為美玟給我的
我再分別對Gossip和Stat去合併
所以只各讀取一個檔
'''
t0 = time()
Gossip = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Gossiping.xlsx",encoding = 'utf-8')
Stat = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Statistics.xlsx",encoding = 'utf-8')
#Stop_word = pd.read_csv("D:\Anaconda3\mycode\MLlab\stopwords_zh.csv",encoding = 'big5',header = None)
print("耗時(秒)=",time()-t0)
#%%整理資料檔
'''
先將Gossiping和Statistics轉成0和1，方便之後的運算
然後將兩個資料合併
因為id不重要，丟掉它
長度太小也沒有意義，我設定是:不到20字都丟掉
'''
Gossip["board"] = Gossip["board"].str.replace("Gossiping","0")#將單字轉成0比較方便
Stat["board"] = Stat["board"].str.replace("Statistics","1")#將單字轉成1比較方便
Combine_GS = Gossip.append(Stat)#將兩dataframe合併
Combine_GS_without_id = Combine_GS[["board","article"]]#丟掉id
Combine_GS_without_id["art_length"] = Combine_GS_without_id["article"].str.len()#新增文章長度
df = Combine_GS_without_id[Combine_GS_without_id["art_length"]>=20]#丟掉長度不到20個字的
df = df[["article","board"]]#交換第一行和第二行
#%%讀取統計詞庫
#國家教育研究院的 雙語詞彙學術名詞暨辭書資訊網
#http://terms.naer.edu.tw/download/466/
Stat_vocab = pd.read_excel("D:\Anaconda3\mycode\MLlab\stat_vocab.xlsx",encoding = 'utf-8')
vocab_eng = Stat_vocab["英文名稱"]
vocab_ch = Stat_vocab["中文名稱"]
stat_dict = np.append(vocab_ch,vocab_eng)
#%%
t0 = time()
#所有句子斷詞後再合併存入articles
#words = jb.cut(text)
jb.set_dictionary("D:\Anaconda3\mycode\MLlab\dict.txt.big.txt")#使用繁體+簡體中文詞庫
jb.load_userdict(stat_dict) #加入我的統計詞庫

articles = []
for text in df["article"]:
    articles.append(jb.cut(text,cut_all=False))#直接返回list
    
new_articles = []
for article in articles:
    article = [x for x in article if (x != '' and x!= ' 'and type(x)!='str')]
    article = np.array(article)
    new_articles.append(article)
df["new_article"] = np.array(new_articles)
print("耗時(秒)=",time()-t0)
#%%
'''
創造一個字典，紀錄每個字詞用的次數
'''
from collections import Counter
counts = Counter()
for new_article in new_articles:
    for word in new_article :
        counts[word]+=1
#這裡的counts是dictionary，紀錄了使用的字詞和次數
#%%
'''
排序由多到少，然後將每個字詞給一個數字去對應
'''
word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:10])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
#將所有字對應到一個數字
'''
順便創一個可以將數字返回文字的字典
'''
int_to_word = dict(zip(word_to_int.values(), word_to_int.keys()) ) 
#%%
'''
將1-hot encoding的類別資料轉回原本的類別
'''
def re1hot(c):
    length = len(c)
    for i in range(length):
        if c[i]!=0:
            c = np.int(i)
            if c == 0:
                return("Gossiping")
            else:
                return("Statistics")
#%%
'''
轉回去拉拉拉
'''
mapped_int = []
pbar2 = pyprind.ProgBar(8038,
                       title='Map ints to reviews')
X_test =X_test.reshape(len(X_test),300)
X_test_list = list(X_test)
for article in X_test_list:
    mapped_int.append([int_to_word[word] for word in article if word != 0])
    pbar2.update()

#%%
