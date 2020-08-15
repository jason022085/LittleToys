# -*- coding: utf-8 -*-
# %%讀取檔案+設定jieba
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import jieba as jb
import pandas as pd
import numpy as np

df = pd.read_excel('TWELF論文簡介.xlsx')
Edu_vocab = pd.read_excel("教育學詞庫.xlsx")
vocab_eng = Edu_vocab["英文名稱"]
vocab_ch = Edu_vocab["中文名稱"]
edu_dict = np.append(vocab_ch, vocab_eng)
# jb.set_dictionary("dict.txt.big.txt")  # 使用繁體+簡體中文詞庫
jb.load_userdict(edu_dict)  # 加入我的教育詞庫

# %%準備斷好的詞,將之新增到df的新欄位


def addColumn(df, column_name):
    titles = []
    for text in df[column_name]:
        titles.append(jb.lcut(text, cut_all=False))  # 直接返回list

    split_titles = []
    for title in titles:
        title = [x for x in title if (
            x != '' and x != ' 'and type(x) != 'str')]
        title = np.array(title)
        split_titles.append(title)
    df['split_'+column_name] = np.array(split_titles)
    return 'split_'+column_name


# %%
#new_col = addColumn(df, "title")
#new_col = addColumn(df, "keywords")
new_col = addColumn(df, "abstract")
# %%將每一組斷詞接起來
all_split_word = []
for i in range(df.shape[0]):
    for word in df[new_col][i]:
        all_split_word.append(str(word))

wl_space_split = " ".join(all_split_word)
# %%


def makeImage(text):
    my_mask = np.array(Image.open("key.png"))
    wc = WordCloud(background_color="white",  # 背景顏色
                   mask=my_mask,
                   max_words=200,  # 最大詞數
                   max_font_size=60,  # 顯示字體的最大值
                   # 解決顯示口字型亂碼問題，可進入C:/Windows/Fonts/目錄更換字體
                   font_path="C:/Windows/Fonts/msjhl.ttc",
                   random_state=42,  # 為每一詞返回一個PIL顏色
                   prefer_horizontal=10)  # 調整詞雲中字體水平和垂直的多少
    my_wordcloud = wc.generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


"""
import re
import jieba

s = "I am Cheng Yo."
print("before adding word :",list(jieba.lcut(s)))
#如果要把中間有空格的片語當作一個word的話，用下面這招
jieba.re_han_default = re.compile("([\u4e00-\u9fa5-zA-z0-Z0-9+#&\._% ]+)",re.U)
jieba.add_word("Cheng Yo")
print(" after adding word :",list(jieba.lcut(s)))
"""
# %%
makeImage(wl_space_split)

# %%
print("world cloud has been drawn.")
