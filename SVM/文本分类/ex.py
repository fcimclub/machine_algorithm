# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:18:20 2018

@author: l_cry
"""
import jieba
import jieba.analyse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

tmt_txt=open('./tmt.txt',encoding='utf8')
eng_txt=open('./eng.txt',encoding='utf8')
food_txt=open('./food.txt',encoding='utf8')

tmt_words = jieba.analyse.extract_tags(tmt_txt.read(), topK=30, withWeight=True, allowPOS=('n'))
eng_words = jieba.analyse.extract_tags(eng_txt.read(), topK=30, withWeight=True, allowPOS=('n'))
food_words = jieba.analyse.extract_tags(food_txt.read(), topK=30, withWeight=True, allowPOS=('n'))

key_words=list(set([words for words,if_idf in tmt_words+eng_words+food_words]))
tmt_txt=open('./tmt.txt',encoding='utf8')
eng_txt=open('./eng.txt',encoding='utf8')
food_txt=open('./food.txt',encoding='utf8')
X1=[]
while 1:
    line = tmt_txt.readline()
    if not line:
        break
    X1.append([1 if word in line else 0 for word in key_words])

X2=[]
while 1:
    line = eng_txt.readline()
    if not line:
        break
    X2.append([1 if word in line else 0 for word in key_words]) 

X3=[]
while 1:
    line = food_txt.readline()
    if not line:
        break
    X3.append([1 if word in line else 0 for word in key_words]) 

X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)


X=np.vstack((X1,X2,X3))
y1=np.zeros(len(X1))
y2 = np.ones(len(X2))
y3 = np.zeros(len(X3))+2

y=np.hstack((y1,y2,y3))
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)
clf = svm.SVC(kernel='linear', C=5000)
clf.fit(x_train, y_train)

clf.score(x_train, y_train)
clf.score(x_test, y_test)


test_txt=open('./test.txt',encoding='utf8')
X_t=[]
while 1:
    line = test_txt.readline()
    if not line:
        break
    X_t.append([1 if word in line else 0 for word in key_words])

clf.predict(X_t)



