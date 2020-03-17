#!/usr/bin/env python
# coding: utf-8

# # 載入套件

# In[1]:
'''
電腦如何自己"手寫"數字
GAN!
'''
# In[2]:


#keras function
from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential
from keras.optimizers import Adam

#畫圖
import matplotlib.pyplot as plt
import numpy as np

#%%
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

# In[17]:


class GAN(object):
#初始設定
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)#使用Adam當優化器

        self.G = self.__generator()#呼叫生成網路
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer,metrics=['accuracy'])

        self.D = self.__discriminator()#呼叫辨別網路
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
#生成網路的建置
    def __generator(self):
        """ Declare generator """

        model = Sequential()
        model.add(Dense(256, input_dim = 100))#這裡餵進100維的向量
        model.add(LeakyReLU(alpha=0.2))  # 使用 LeakyReLU 激活函數
        model.add(BatchNormalization(momentum=0.8))  # 使用 BatchNormalization 優化
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))#28*28*1
        model.add(Reshape((self.width, self.height, self.channels)))#這個轉換很重要，從向量轉回圖片
        model.summary()

        return model

# 辨別網路的建置
    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(Flatten(input_shape=self.shape))#拉平
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int((self.width * self.height * self.channels)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))#輸出 1 or 0
        model.summary()

        return model

# 將兩個網路串在一起
    def __stacked_generator_discriminator(self):

        self.D.trainable = False #??

        model = Sequential()
        model.add(self.G)#網路中的網路，酷吧！
        model.add(self.D)

        return model

#訓練GAN網路
    def train(self, X_train, epochs=10000, batch = 32, save_interval = 100):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - batch/2)
            legit_images = X_train[random_index : random_index + int(batch/2)].reshape(int(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, size = (int(batch/2), 100)) #常態分布的資料餵給生成網路
            syntetic_images = self.G.predict(gen_noise)#生成網路產生一些圖

            x_combined_batch = np.concatenate((legit_images, syntetic_images))#(真圖、假圖)
            y_combined_batch = np.concatenate((np.ones((int(batch/2), 1)), np.zeros((int(batch/2), 1)))) #(16個1,16個0)

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)#分批訓練，每一批都調整權重
            #返回批量訓練誤差（如果模型只有單個輸出且沒有評估指標）
            #https://keras.io/zh/models/sequential/

            # train generator

            noise = np.random.normal(0, 1, (batch, 100))  # 添加高斯噪聲 batch筆100維的向量
            y_mislabled = np.ones((batch, 1))#經由G-D網路後，lable是1(判斷正確的意思、騙過辨別網路的意思)
            '''
            -----input------
            np.ones((3, 1))
            -----output-----
            array([[1.],
           [1.],
           [1.]])
            '''
            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0: #每100次訓練後
                self.plot_images(save2file=True, step=cnt)#輸出GAN的生成圖片
            
# 儲存GAN生成的圖像        
    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise,verbose=1)
    
        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


# # 開始訓練
# * 先練辨別網路D
# * 後練生成網路G
# * 再練辨別網路D
# * 往後以此類推

# # 主程式Main

# In[18]:


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()#不需要lable，因為這是Unsupervised Learning

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)


    gan = GAN()
    gan.train(X_train,epochs=10000)
#%%



