# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:58:14 2019

@author: M.Tahir
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import optimizers

import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

dataset=pd.read_csv("heart.csv")


filename='M.Tahir.h5'
def predict(model1,control):
    if (control==1):
        model1=load_model(filename)
    y_pred=model1.predict(X_test)
    y_pred = np.around(y_pred)
    y_test_non_category = [ np.argmax(t) for t in Y_test ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    score, acc = model1.evaluate(X_test, Y_test, batch_size=128)
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    #F1 score hesaplanır
    f1=f1_score(Y_test, y_pred,average='weighted')
    if (control==1):
        print (conf_mat) 
        print(classification_report(Y_test, y_pred)) 
        print("Acc=",acc)      
    return acc,f1

def predict_egitim(model1,control):
    if (control==1):
        model1=load_model(filename)
    y_pred=model1.predict(X_train)

    y_pred = np.around(y_pred)

    y_test_non_category = [ np.argmax(t) for t in Y_train ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]

    score, acc = model1.evaluate(X_train, Y_train, batch_size=128)

    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    f1=f1_score(Y_train, y_pred,average='weighted')
    if (control==1):
        print (conf_mat) 
        print(classification_report(Y_train, y_pred)) 
        print("Acc=",acc)  
        
    return acc,f1

Train, Test= train_test_split(dataset, test_size=0.30, random_state=60)
Train.to_excel("train.xlsx")
Test.to_excel("test.xlsx")
X_train=Train.iloc[:,0:13]
X_test=Test.iloc[:,0:13]
Y_train=Train.iloc[:,13]
Y_test=Test.iloc[:,13]


encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train= encoder.transform(Y_train)
encoder.fit(Y_test)
Y_test= encoder.transform(Y_test)


Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

acc1=0.1
while(acc1<0.65):
    a=2
    start=0
    finish=7
    for k in range(start,finish):
        if (k==0):opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if (k==1):opt=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        if (k==2):opt = optimizers.SGD(lr=0.01, clipnorm=1.)
        if (k==3):opt =keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        if (k==4):opt=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        if (k==5):opt=keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        if (k==6):opt=keras.optimizers.Adagrad(lr=0.01 ,epsilon=None ,decay=0.0)

        model = Sequential()
        aktivasyon="softsign"

        model.add(Dense(16, input_dim=13, activation=aktivasyon))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(48, activation=K.tanh))
        model.add(Dense(64, activation=aktivasyon))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        for i in range(10):

            model.fit(X_train, Y_train, epochs=40, batch_size=64,verbose=0)
            acc,fs=predict(model,0)

            print("i=",i,"k=",k," F score:",fs," Acc=",acc)
            if i==0 and k==start:
                eb=fs
                ebacc=acc
                tempk=k
                model.save("M.Tahir.h5")
            if (fs>=eb):
                if (fs==eb):
                    if (acc>ebacc):
                        ebacc=acc
                        tempk=k
                        model.save("M.Tahir.h5")
                else:
                    eb=fs
                    tempk=k
                    model.save("M.Tahir.h5")

        print ("En Yüksek F Skoru:",eb, "K Değeri=",tempk, "En Yüksek Acc",ebacc)
        print("Test Sonuç")

    acc1,fs1=predict(model,1)
    predict_egitim(model,1)
