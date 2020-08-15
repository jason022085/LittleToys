# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 01:05:10 2019

@author: 皇甫承佑
嘗試用r"regular expression"找到名字
"""

# %%
import re
import os
import numpy as np
import pandas as pd
import PyPDF2
import pprint

# %%


def Pdf2text(path):
    PDF = PyPDF2.PdfFileReader(path)
    # 列出PDF檔案中所有的文字
    text = ""
    for page in PDF.pages:
        text += page.extractText()
    return text


def Find_name_en(article):
    # =========================================================================
    '''
    以下檔案請放同一資料夾中，並且編碼方式須為utf-8
    '''
    # =========================================================================
    # 這邊是照著Referenve的姓名格式打的，只能找到Reference的姓名。
    target_en1 = r"[a-zA-Z]{1,}, [A-Z]{1}\. [A-Z]{1}\."  # Carrier, L. M.
    target_en2 = r"[a-zA-Z]{1,}, [A-Z]{1}\."  # Chajut, E.
    target_en3 = r"[a-zA-Z]{1,} [A-Z]{1,2}\."  # AAAAA, AB.
    # 第3個這個是拿來對付A qualitative review of the design thinking framework in health professions education的
    target_ch = r"[a-z]{1,}, [a-z]\.-[a-z]\."  # Wu, J.-Y.
    # 打開文檔
    path = "D:\\Anaconda3\\mycode\\FindName2020\\PDFs\\"
    data = Pdf2text(path+article) #這邊需要自訂
    # 外國人英文名
    En_name = []
    for text in data:
        # temp_en1 = re.match(target_en1, text, re.I | re.M)
        # search才能搜尋整個文件，match法只要開頭不是就搜尋不到
        temp_en1 = re.search(target_en1, text, re.I | re.M)
        temp_en2 = re.search(target_en2, text, re.I | re.M)
        temp_en3 = re.search(target_en3, text, re.I | re.M)
        if temp_en3 != None:
            En_name.append(temp_en3.group())
        if temp_en2 != None:
            En_name.append(temp_en2.group())
        if temp_en1 != None:
            En_name.append(temp_en1.group())  # 加上group才變成搜尋到的字串，否則為整體re的object
    # 本國人的英文名
    Ch_name = []
    for text in data:
        temp_ch = re.search(target_ch, text, re.I | re.M)
        if temp_ch != None:
            Ch_name.append(temp_ch.group())

    All_name = En_name + Ch_name
    All_name_list = list(set(All_name))  # 解決了重複的問題
    # 解決包含的問題
    Sub_name_list = []  # Carrier, L. M. 與 Carrier, L. 應該被算做同一人
    for i in range(len(All_name_list)):
        for j in range(i+1, len(All_name_list)):
            if All_name_list[j].find(All_name_list[i]) != -1:
                Sub_name_list.append(All_name_list[i])

    # 恰好的名稱
    Fit_name = set(All_name_list)-set(Sub_name_list)
    Fit_name_list = list(Fit_name)
    # 返回一列表
    return Fit_name_list


def build_table(path):
    df = pd.DataFrame()
    for i in os.listdir(path):
        Name = Find_name_en(i)
        df = df.append([i, Name])
    return df


# %%
path = "D:\\Anaconda3\\mycode\\FindName2020\\PDFs"
df_op = build_table(path)
df_op.to_excel("test.xlsx")
# %%
