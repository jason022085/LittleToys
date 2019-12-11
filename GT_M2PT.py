# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 23:07:59 2019

@author: USER
"""

def genarate(n):
    import numpy as np
#    import math as mt
    import random as rd
#    n = input('請輸入總數：')
#    n = int(n)
    while(n <= 2):#防呆
        n = input('請輸入比2大的數字當樣本總數：')
        n = int(n)
        
    S = np.zeros(n-2)#建立都是0的array
    S = np.append(S,1)#加入兩個1
    S = np.append(S,1)
    S = S.tolist()#轉成list
    rd.shuffle(S)#將列表隨機排序
    count = 0#計算測驗次數
    S = [int(i) for i in S]#浮點數轉成整數
        
    return S,count
#%%
def step1(S):
    n = len(S)
    if n == 0 or n == 1:
        big2 = 0
        lit2 = 0        
        #如果n太小，k值會變0或負的，big2和lit2會變無意義，所以這邊拉出來定義
    else:
        for i in range(n):
            if pow(2,i) <= len(S) and len(S) < pow(2,i+1):
                k = i 
                big2 = pow(2,k)#B
                lit2 = pow(2,k-1)#L           #以上步驟可確認左子節點的數值
    S_temp = S
    S = []
    if n == 1:
        locat = "o"
        S = S
        
    elif n != big2 :
        if 1 in S_temp[big2:n]:
            S = S_temp[big2:n]            
            locat = "R"
            S_LS = S_temp[0:big2]
                        
        else:
            S = S_temp[0:big2]
            locat = "L"
            S_LS = []
                                  
    elif n == big2 :
        if 1 in S_temp[lit2:n]:
            S = S_temp[lit2:n]
            locat = "R"
            S_LS = S_temp[0:lit2]
            
        else:
            S = S_temp[0:lit2]
            locat = "L" 
            S_LS = []
          
    return S,S_LS,locat
#%%
def test1(S,count):
    if(len(S) > 1):
        
        S,S_LS,locat = step1(S)
        print(S,locat)
        
        count = count +1
        if S_LS == []:
            return test1(S,count)
        else:
            return test1(S,count),S_LS
    else:
        print("END")
        print("所需測驗次數=",count)    
#%%
def step2(S):
#    import numpy as np
#    import math as mt
    n = len(S)
    if n == 0 or n == 1:
        big2 = 0
        lit2 = 0
        #如果n太小，k值會變0或負的，big2和lit2會變無意義，所以這邊拉出來定義
    else:
        for i in range(n):
            if pow(2,i) <= len(S) and len(S) < pow(2,i+1):
                k = i 
                big2 = pow(2,k)
                lit2 = pow(2,k-1)           
    #以上步驟可確認左子節點的數值
    S_temp = S
    S = []
#    S_nd = []
    if n == 1:
        locat = "o"
        S = S
        
    elif n != big2 :
        if 1 in S_temp[0:big2]:
            S = S_temp[0:big2]            
            locat = "L"            
        else:
            S = S_temp[big2:n]
            locat = "R"
            
            
    elif n == big2 :
        if 1 in S_temp[0:lit2]:
            S = S_temp[0:lit2]
            locat = "L"
            
        else:
            S = S_temp[lit2:n]
            locat = "R" 
            
    return S,locat
#%%
def test2(S,count):
    if(len(S) > 1):
        S,locat = step2(S)
        print(S,locat)
        
        count = count +1
        return test2(S,count)
    else:
        print("END")
        print("所需測驗次數=",count)          
#%%
def M2PT():
    S,count = genarate(2)
    print("原數列:",S)

    X,S_LS = test1(S,count)
    print("---------------------")
#step3
    if S_LS != [] and 1 in S_LS:   
        print("測出左節點有陽性:",S_LS)
        if S_LS == [1] or len(S_LS) == len(S)-1:
            test1(S_LS,count)
        else:
            test1(S_LS,count+1)
    elif 1 not in S_LS:   
        S,X,locat = step1(S)
        print("測出左節點無陽性:",S)
        test2(S,count)
#%%
again = "y"
while(again == 'Y' or again == 'y'):
    M2PT()
    again = input("輸入y可重新開始，否則結束程式：")
print("瀛海中學數學科科展感謝您~")
    