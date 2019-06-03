# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:55:02 2019

@author: CY H.F.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #專業繪圖套件
from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,Dropout #flatten是要將矩陣拉平成向量
from keras.layers import Conv2D,MaxPooling2D,noise,BatchNormalization
from keras.optimizers import SGD,Adam
import keras as kr
import tensorflow as tf

#%%下載data和ˋmodel
D_T = np.load('D:\Anaconda3\程式碼\chart\DT200_6.npz')
#%%
X = D_T['data']
y = D_T['target']
#%%
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7,stratify=y)
print("data分割完成")

X_train =X_train.reshape(len(X_train),200,200,1)
X_test =X_test.reshape(len(X_test),200,200,1)
from keras.utils import np_utils#1-hot encoding
y_train = np_utils.to_categorical(y_train,6)
y_test = np_utils.to_categorical(y_test,6)
#%%
model = Sequential()#C-P-C-P-Dense-Dense

model.add(Conv2D(64,(3,3),padding='same',input_shape=(150,150,1)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(1,1),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256,(1,1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(lr = 0.0003), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])#組裝神經網路
#CNN到此已完成!
model.summary()
print("model建立完成")
#%%
history = model.fit(X_train,y_train,batch_size=16,epochs=32)
(err,acc) = model.evaluate(X_test,y_test)
print("(err,acc) = ",(err,acc))
#%%
model.save('D:\Anaconda3\程式碼\chart\model_chart_cnn85.h5')
print("model儲存完成")
#%%以後從這裡執行就好
#
#
#
#
#
#%%
D_T = np.load('D:\Anaconda3\mycode\chart\DT200_6.npz')
model = kr.models.load_model('D:\Anaconda3\mycode\chart\model_chart_cnn91.h5')
#model = kr.models.load_model('D:\Anaconda3\mycode\chart\model_cnn7_888.h5')
#%%
X = D_T['data']
y = D_T['target']

from sklearn.model_selection import train_test_split
X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print("data分割完成")

X_train =X_train.reshape(len(X_train),200,200,1)
X_test =X_test.reshape(len(X_test),200,200,1)
from keras.utils import np_utils#1-hot encoding
y_train = np_utils.to_categorical(y_train,6)
y_test = np_utils.to_categorical(y_test,6)
#%%
predict = model.predict_classes(X_test)
rad = np.random.randint(0,len(X_test))
print("第",rad,"張")
print('0=長條,1=直方,2=圓餅,3=泡泡,4=散點,5=折線圖')
print("預測為:",predict[rad])
print("實際為:",y_test[rad])
image = X_test[rad].reshape(200,200)
plt.figure(num = 'haha' ,figsize=(6,6))
plt.imshow(image,cmap='gray')
#%%因為是直接讀取model，所以沒有訓練過成了，這裡不能執行
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%畫出模型= =
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_layer_names = False,show_shapes=True)
#%%
model.summary()
(err,acc) = model.evaluate(X_test,y_test)
print(acc)
