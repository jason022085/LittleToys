# -*- coding: utf-8 -*-
"""
Created on Mon May 13 02:43:13 2019

@author: 半夜3點的皇甫
"""
#%%套件
import speech_recognition as sr
#%%收麥克風的音
r=sr.Recognizer() 
with sr.Microphone() as source:
    print("Please wait. Calibrating microphone...") 
    #listen for 5 seconds and create the ambient noise energy level 
    r.adjust_for_ambient_noise(source, duration=5) 
    print("Say something!")
    audio = r.listen(source)

#%%讀取音檔wav和flac可以讀進去

r = sr.Recognizer()#開錄音機 
with sr.AudioFile( r"C:\Users\USER\Desktop\test.wav") as source:
    audio = r.record(source)

#%%google語音辨識(只有flac可被接受)
# recognize speech using Google Speech Recognition 
try:
    print("Google Speech Recognition thinks you said:")
    print(r.recognize_google(audio, language="zh-TW"))
    seten = r.recognize_google(audio, language="zh-TW")
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))
#%%google翻譯
from googletrans import Translator

translator = Translator()
# src來源語言，dest要翻譯語言，如果要找其他語言可以參考說明文件
#seten = input("輸入一句話:")
trans = translator.translate(str(seten), src = "zh-TW", dest = "zh-TW")
text = trans.text
print (text)
#%%結巴斷詞
import jieba as jb
'''
這裡有個問題是本語料庫是針對中文一般用語
而非統計用語，所以斷詞會斷的很爛
'''
#words = jb.cut(text)
words = jb.lcut(text,cut_all=False)#直接返回list
print(words)
'''
秀一下詞性的功能
input:
    
import jieba.posseg as pseg
words2 = pseg.cut("我是鋼鐵人")
for word, flag in words2:
    print((word, flag))
output:
    
('我', 'r')
('是', 'v')
('鋼', 'n')
('鐵人', 'n')
'''
#%%判斷答題正確
KW = ["獨立性" ,"常態性" , "同質性"]
if (KW[0] and KW[1] and KW[2]) in words:
    print ("答對了！")
else:
    print("答錯了。")
#%%ML該用在哪裡?
