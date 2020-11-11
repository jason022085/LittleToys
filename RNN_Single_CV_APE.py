# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 22:06:58 2020

@author: 皇甫承佑

資料前處理：加權 -> 排名 -> 最大最小歸一化
"""

# %% Set random seed
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.layers import Input, Dropout, concatenate  # 功能型
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding  # 真正的神經網路層
import tensorflow
from time import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from numpy.random import seed
seed(2020)

tensorflow.random.set_seed(2020)
# %%
df_train = pd.read_excel("訓練集_混成式課程特徵.xlsx")
df_test = pd.read_excel("測試集_混成式課程特徵.xlsx")
# get  y
y_train = df_train['Fail']
y_train_onehot = to_categorical(y_train)
y_test = df_test['Fail']
y_test_onehot = to_categorical(y_test)

# %%
P = ["Name","HW1","HW2","HW3","HW4","Score_total","Fail"]

B = ["Age","Male","Degree","NETtime_coding","FBage","Selfaca_true"] #Name先借放在這

A = ["Computational_self_concept","Worth_of_statistics","Test_and_class_anxiety",\
     "Explain_by_eng_anxiety","Fear_of_statistics","Fear_of_asking_for_help",\
     "software_anxiety","Data_analysis_anxiety","Interpretation_anxiety"]

E = ['Post_P1', 'Post_P2', 'Post_P3', 'Post_P4',\
     'Comment_P1', 'Comment_P2', 'Comment_P3', 'Comment_P4',\
     'StaPercent_P1', 'StaPercent_P2', 'StaPercent_P3', 'StaPercent_P4']
# %%
wanted_col = A+P+E
wanted_col = set(wanted_col)


all_col = df_train.columns
all_col = set(all_col)

drop_col = all_col - wanted_col
drop_col = list(drop_col)

for col in drop_col:
    df_train = df_train.drop([col], axis=1)
    df_test = df_test.drop([col], axis=1)
# %% feature engineering
negtive_corr_col = ['Selfaca_true', 'FB_club', 'software_anxiety', 'FBlive1', 'Degree',
                    'Age', 'software_anxiety', 'NETtime_coding',
                    'StaPercent_P3', 'StaPercent_P2', 'StaPercent_P1',
                    'Reply_P3', 'Reply_P2', 'Reply_P1',
                    'Post_P4', 'Post_P3', 'Post_P2', 'Post_P1',
                    'Comment_P4', 'Comment_P3', 'Comment_P2', 'Comment_P1',
                    'Like_P4', 'Like_P3', 'Like_P2', 'Like_P1',
                    'HW4', 'HW3', 'HW2', 'HW1', 'Final_project']


def feature_engineering(df_train, feature_ranking=True, turn_negative_corr=True):
    # negitive
    if turn_negative_corr == True:
        for col in negtive_corr_col:
            try:
                df_train[col] = -df_train[col]
            except:
                None
    # feature engineering: rank
    if feature_ranking == True:
        all_col = list(df_train.keys())
        all_col.remove("Fail")
        all_col.remove("Score_total")
        for col in all_col:
            df_train[col] = df_train[col].rank(
                method="max", na_option='bottom')  # NA設為最低rank

    # feature engineering: MM scalar
    X_P04 = df_train.drop(['Name', 'Score_total', 'Fail'], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X_P04)
    X_P04 = scaler.transform(X_P04)
    # % get X from P0 to P4
    X_P04 = pd.DataFrame(X_P04)  # 5 + 4 +2
    X_P03 = X_P04.iloc[:, :21]  #
    X_P02 = X_P04.iloc[:, :17]  #
    X_P01 = X_P04.iloc[:, :13]  #
    X_P0 = X_P04.iloc[:, :1]  # 19
    return X_P0, X_P01, X_P02, X_P03, X_P04


# %%
X_train_P0, X_train_P01, X_train_P02, X_train_P03, X_train_P04 = feature_engineering(
    df_train)
X_test_P0, X_test_P01,  X_test_P02,  X_test_P03,  X_test_P04 = feature_engineering(
    df_test)


def X_reshape(X):
    X = np.asarray(X)
    X = X.reshape((X.shape[0], X.shape[1]))  # 人數,特徵數,每個特徵的維度
    return X


X_train_P0 = X_reshape(X_train_P0)
X_train_P01 = X_reshape(X_train_P01)
X_train_P02 = X_reshape(X_train_P02)
X_train_P03 = X_reshape(X_train_P03)
X_train_P04 = X_reshape(X_train_P04)

X_test_P0 = X_reshape(X_test_P0)
X_test_P01 = X_reshape(X_test_P01)
X_test_P02 = X_reshape(X_test_P02)
X_test_P03 = X_reshape(X_test_P03)
X_test_P04 = X_reshape(X_test_P04) 
# %%
"""
建立模型與交叉驗證，用以調整神經網路架構和超參數
"""
# 自訂網路


def Create_modelP1():
    # 連接網路層
    shape_in = 13
    x = Input(shape=(shape_in,))
    h1 = Embedding(128, 16)(x)
    h2 = LSTM(32, kernel_initializer='orthogonal',
              kernel_regularizer=l2(0.001),
              dropout=0.1, return_sequences=False)(h1)
    y = Dense(2, kernel_initializer="glorot_normal", activation="softmax")(h2)
    # 組裝神經網路
    model = Model(x, y)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    # model.summary()
    return model


def Create_modelP2():
    # 連接網路層
    shape_in = 17
    x = Input(shape=(shape_in,))
    h1 = Embedding(128, 16)(x)
    h2 = LSTM(32, kernel_initializer='orthogonal',
              kernel_regularizer=l2(0.001),
              dropout=0.1, return_sequences=False)(h1)
    y = Dense(2, kernel_initializer="glorot_normal", activation="softmax")(h2)
    # 組裝神經網路
    model = Model(x, y)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    # model.summary()
    return model


def Create_modelP3():
    # 連接網路層
    shape_in = 21
    x = Input(shape=(shape_in,))
    h1 = Embedding(128, 16)(x)
    h2 = LSTM(32, kernel_initializer='orthogonal',
              kernel_regularizer=l2(0.001),
              dropout=0.1, return_sequences=False)(h1)
    y = Dense(2, kernel_initializer="glorot_normal", activation="softmax")(h2)
    # 組裝神經網路
    model = Model(x, y)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    # model.summary()
    return model


def Create_modelP4():
    # 連接網路層
    shape_in = 25
    x = Input(shape=(shape_in,))
    h1 = Embedding(128, 16)(x)
    h2 = LSTM(32, kernel_initializer='orthogonal',
              kernel_regularizer=l2(0.001),
              dropout=0.1, return_sequences=False)(h1)
    y = Dense(2, kernel_initializer="glorot_normal", activation="softmax")(h2)
    # 組裝神經網路
    model = Model(x, y)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    # model.summary()
    return model
# %%


def Plot_Training(history, showACC=False, showLOSS=True):
    if showACC:
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    if showLOSS:
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


# %%交叉驗證法
def CrossValidate_Model(Create_modelP0, X_train_P0, times=9, k=5, EPOCHS=16, plot=False):
    P_acc, P_f1s, P_pre, P_rec = [], [], [], []
    time_spent_all = []
    y_pred_list_all = []
    for i in range(times):
        kfold = KFold(n_splits=k, shuffle=False)  # 分割資料變成k等分的設定，不打亂順序
        ksplit = kfold.split(X_train_P0,  y_train)  # 這是分好的k等分(是一種generator)
        precision = []
        recall = []
        accuracy = []
        f1s = []
        time_start = time()
        y_pred_list = []
        for j in range(k):

            model = Create_modelP0()
            train_no = next(ksplit)
            train_no, valid_no = train_no[0], train_no[1]
            history = model.fit(X_train_P0[train_no], y_train_onehot[train_no],
                                validation_data=(
                                    X_train_P0[valid_no], y_train_onehot[valid_no]),
                                batch_size=1, epochs=EPOCHS, verbose=0)  # verbose調成1
            if plot == True:
                Plot_Training(history)
            y_pred = model.predict(X_train_P0[valid_no])
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_train_onehot[valid_no], axis=1)

            pre = precision_score(
                y_true, y_pred, average="binary", zero_division=1)
            rec = recall_score(
                y_true, y_pred, average="binary", zero_division=1)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=1)
            acc = accuracy_score(y_true, y_pred)

            precision.append(pre)
            recall.append(rec)
            accuracy.append(acc)
            f1s.append(f1)
            y_pred_list.extend(y_pred)
        time_spent_all.append(round(time()-time_start, 4))
        y_pred_list_all.append(y_pred_list)
        mean_acc = np.mean(accuracy)
        mean_pre = np.mean(precision)
        mean_rec = np.mean(recall)
        mean_f1s = np.mean(f1s)

        P_acc.append(mean_acc)
        P_f1s.append(mean_f1s)
        P_rec.append(mean_rec)
        P_pre.append(mean_pre)
        print("mean accuracy= ", round(mean_acc, 4), ";mean precision= ", round(mean_pre, 4),
              ";mean recall= ", round(mean_rec, 4), ";mean f1= ", round(mean_f1s, 4))
    return P_acc, P_f1s, P_pre, P_rec, time_spent_all, y_pred_list_all
# %%


def Record_Metrics(EPOCHS=16):
    print("P1")
    P1_acc, P1_f1s, P1_pre, P1_rec, P1_time, P1_pred = CrossValidate_Model(
        Create_modelP1, X_train_P01, EPOCHS=EPOCHS)  # P1

    print("P2")
    P2_acc, P2_f1s, P2_pre, P2_rec, P2_time, P2_pred = CrossValidate_Model(
        Create_modelP2, X_train_P02, EPOCHS=EPOCHS)  # P2

    print("P3")
    P3_acc, P3_f1s, P3_pre, P3_rec, P3_time, P3_pred = CrossValidate_Model(
        Create_modelP3, X_train_P03, EPOCHS=EPOCHS)  # P3

    print("P4")
    P4_acc, P4_f1s, P4_pre, P4_rec, P4_time, P4_pred = CrossValidate_Model(
        Create_modelP4, X_train_P04, EPOCHS=EPOCHS)  # P4

    P_all_acc = np.concatenate([P1_acc, P2_acc, P3_acc, P4_acc])
    P_all_pre = np.concatenate([P1_pre, P2_pre, P3_pre, P4_pre])
    P_all_rec = np.concatenate([P1_rec, P2_rec, P3_rec, P4_rec])
    P_all_f1s = np.concatenate([P1_f1s, P2_f1s, P3_f1s, P4_f1s])
    P_all_time = np.concatenate([P1_time, P2_time, P3_time, P4_time])
    P_all_pred = np.concatenate([P1_pred, P2_pred, P3_pred, P4_pred])

    df_op = pd.DataFrame([P_all_acc, P_all_f1s, P_all_pre, P_all_rec, P_all_time, P_all_pred], index=[
                         "Accuracy", "F1", "Precision", "Recall", "Time", "Prediction"])
    return df_op


# %% 視覺化檢查
Create_modelP4().summary()
CrossValidate_Model(Create_modelP1, X_train_P01, times=1, EPOCHS=32)
# %%


def main():

    df_32epochs = Record_Metrics(32)

    df_32epochs.to_excel("實驗數據/RNN_32_APE.xlsx")

