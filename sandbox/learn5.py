import numpy as np
import timeit
from sklearn import svm
import matplotlib.pyplot as plt
import struct       #modun dung de dinh dạng ban ghi nhi phan , giai nen du lieu #https://www.geeksforgeeks.org/struct-module-python/
import timeit
import pickle
from skimage import io


TRAIN_ITEMS = 60000
TEST_ITEMS = 10000
# train-images-idx3-ubyte: đào tạo tập hình ảnh
# đào tạo-nhãn-idx1-ubyte: nhãn tập huấn luyện
# t10k-images-idx3-ubyte: kiểm tra tập hình ảnh
# t10k-labels-idx1-ubyte: nhãn thiết lập thử
#Tập huấn luyện có 60000, bài kiểm tra 10000

import numpy as np
import timeit
from sklearn import svm
import struct

TRAIN_ITEMS = 60000
TEST_ITEMS = 10000

def loadMnistData():
    mnist_data = []
    for img_file,label_file,items in zip(['../data/train-images-idx3-ubyte','../data/t10k-images-idx3-ubyte'],
                                   ['../data/train-labels-idx1-ubyte','../data/t10k-labels-idx1-ubyte'],
                                   [TRAIN_ITEMS, TEST_ITEMS]):
        data_img = open(img_file, 'rb').read()
        data_label = open(label_file, 'rb').read()

        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset)
       
        offset += struct.calcsize(fmt)
      
        image_size = height * width
      
        fmt = '>{}B'.format(image_size)
     
        if items > img_number:
            items = img_number
        images = np.empty((items, image_size))
        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
           
            images[i] = images[i]/256
            offset += struct.calcsize(fmt)


        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)
        # print('magic number is {} and label number is {}'.format(magic_number, label_number))
       
        offset += struct.calcsize(fmt)
        #B means unsigned char
        fmt = '>B'
       
        if items > label_number:
            items = label_number
        labels = np.empty(items)
        for i in range(items):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)
        
        mnist_data.append((images, labels.astype(int)))
    return mnist_data

def forwardWithSVM():
    start_time = timeit.default_timer()
    training_data, test_data = loadMnistData()
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    train_time = timeit.default_timer()
    # print('gemfield train cost {}'.format(str(train_time - start_time) ) )
    # test
    print('Begin the test...')
    predictions = [int(a) for a in clf.predict(test_data[0])]
    print("predictions:", predictions)
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    print("%s of %s values correct." % (num_correct, len(test_data[1])))
    test_time = timeit.default_timer()
    # print('gemfield test cost {}'.format(str(test_time - train_time) ) )
forwardWithSVM()

