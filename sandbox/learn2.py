import csv as csv
import pandas as pd
import numpy as np

from sklearn import datasets, svm, metrics

#Đây là một thư viện dùng để vẽ đồ thị.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#http://viet.jnlp.org/home/nguyen-van-hai/nghien-cuu/mlearning/building-machine-learning-system-using-python/chng-1-bt-u-vi-python/ve-do-thi-su-dung-mathplotlib
import matplotlib

import math

# Đọc file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Lấy từ cột 1 trở đi
data_train=train.iloc[:,1:].values		
# print(" data_train= train.iloc[:,1:ư.values: ", data_train)

# Lấy cột label
label_train=train.iloc[:,0].values		
# print("label_train= train.iloc[:,0].values:  ", label_train)

test=test.values
# print("test=test.values: ", test)
	
data_train[data_train>0]=1		#thay các sô >0 thành 1 . vd: 254 = 1
test[test>0]=1					#tahy các số >0 thành 1 trong file test
print("data_train[0]:" , data_train[0])
# print("dât_train: ",dât_train)
# print("label_train: ", label_train)

# Tạo một cái phân loại: Phân loại máy vector hỗ trợ 
classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
# #classifier = svm.SVC()
# print("classifier: ", classifier)


# Cho nó hóc
classifier.fit(data_train, label_train)


predicted = classifier.predict(test)		#ra nhãn của test	
print("predicted: ", predicted)			

df=pd.DataFrame(predicted)
print("df: ", df)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv',header=True)
print("predicted: ", predicted)