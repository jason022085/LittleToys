# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:38:57 2018

@author: USER
"""
#每張卡有n個圖案,每個圖案出現n次
'''
設定最一開始的數字
'''
print('皇甫幫你製作塔寶遊戲~\n規則:每張卡有n個圖案,每個圖案出現n次')
print('問題:1.需要多少張卡片?2.如果圖案由數字1開始,能否將卡片做出來?')
n = int(input('請設定n='))
num_pic = n**2-n+1 #圖案總數
num_card = n*(n-1)+1#卡片總數
sort_pic = list(range(num_pic))#設定順序
sort_pic = [i+1 for i in sort_pic] 
print('現在的圖案總數=',num_pic)
print('現在的卡片總數=',num_card)
print('全部的圖案分別是',sort_pic,'\n')

'''
設定中間的矩陣
'''
matlist = []#建構第一區塊的矩陣
for N in range(n): 
    mattemp = []
    for i in range(n-1):
        temp = []
        for j in range(n-1):
            temp.append(j+n*(i+1)+(1-i))
        mattemp.append(temp)
    matlist.append(mattemp)  
'''
輸出基本元素
''' 
print('第1張卡片(基本元素):',list(range(1,n+1)),'\n')   
'''
輸出第一個區塊
'''
import numpy as np ##呼叫資料庫
matlist[0] = list(matlist[0])

for i in range(n-1):
    matlist[0][i].insert(0,1) #加入基本元素

matlist[0] = np.array(matlist[0])
matlist[0] = matlist[0].reshape(n-1,n)
#print('第 1 區塊:\n',matlist[0],'\n')#完整的第一區塊

'''
輸出第二個區塊
'''
for i in range(1,n):
    matlist[i] = np.transpose(matlist[i])
    matlist[i] = matlist[i].tolist() #先將數組化為列表才能插入值

for i in range(1,n):
    for j in range(n-1):
        matlist[i][j].insert(0,i+1)

for i in range(n):
    matlist[i] = np.array(matlist[i])
    matlist[i] = matlist[i].reshape(n-1,n)    




#print('第 2 區塊:\n',matlist[1],'\n')#完整的第二區塊

'''
製作其他區塊
'''
for i in range(n-1):
    mattemp[i] = mattemp[i] + mattemp[i] #製作拉丁方陣需要用到這個
del mattemp[0]


for i in range(2,n):
    for j in range(n-1):
        for k in range(2,n):
            matlist[i][j][k] = mattemp[k-2][j+i-1]

'''
輸出所有卡片
'''  
count = 2          
for i in range(0,n):
    for j in range(n-1):
        print('第',count,'張卡片:\n',matlist[i][j],'\n')
        count += 1

over = int(input('輸入任何按鍵以結束程式'))

