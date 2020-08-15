#中文斷詞
import pandas as pd
import numpy as np
import jieba
#使用繁體+簡體中文詞庫
jieba.set_dictionary("D:\Anaconda3\mycode\MLlab\dict.txt.big.txt")
#加入我的統計詞庫
Stat_vocab = pd.read_excel("D:\Anaconda3\mycode\MLlab\stat_vocab.xlsx",encoding = 'utf-8')
vocab_eng = Stat_vocab["英文名稱"]
vocab_ch = Stat_vocab["中文名稱"]
stat_dict = np.append(vocab_ch,vocab_eng)
jieba.load_userdict(stat_dict)
#斷詞
text = "大家好，我今天用迴歸分析有顯著效果"
print('斷詞前:',text)
print('斷詞後:', '|'.join(jieba.cut(text, cut_all=False, HMM=True)))
#再移除停止詞
Stop_word = pd.read_csv("D:\Anaconda3\mycode\MLlab\stopwords_zh.csv",encoding = 'big5',header = None)
stopwords = list(Stop_word[0])
words = [w for w in jieba.lcut(text, cut_all=False, HMM=True) if w not in stopwords]
print('移除停止詞後:', '|'.join(words))
