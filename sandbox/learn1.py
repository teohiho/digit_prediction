import csv as csv
import pandas as pd
import numpy as np
import tensorflow as tf


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 
train_x = train.drop('label', axis=1)	#xoa cot label
train_y = pd.DataFrame(np.zeros([len(train['label']), 10]), columns=[0,1,2,3,4,5,6,7,8,9])	#bang cot 10, hang 42000

# print("train: ", train)
# print("train_x: ", train_x)
# print("train_x.shape", train_x.shape)
# print("train_y: ", train_y)
# print("train_y.shape: ",train_y.shape)

for idx in train.index:
    col = train.ix[idx, 'label']
    train_y.ix[idx, col] = 1
    # print(idx) #--> 41999
    # print("col", col)		#lay gia tri o cot label col 1, col 0,..
    # print("train_y.ix[idx, col]: ", train_y.ix[idx, col])    # Dung ra mot tra tran, cot col, ngang idx

   	
# print("train.ix", train.ix)
# print("train_y", train_y)	# Dung ra mot tra tran, cot col, ngang idx
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28 * 28 ])
# print(x)
y_true = tf.placeholder(tf.float32, shape=[None, 10])


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# print("W::",W)
# print("b::",b)
sess.run(tf.global_variables_initializer())