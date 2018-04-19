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

def loadMnistData():
    mnist_data = []
    for img_file,label_file,items in zip(['data/train-images-idx3-ubyte','data/t10k-images-idx3-ubyte'],
                                   ['data/train-labels-idx1-ubyte','data/t10k-labels-idx1-ubyte'],
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
           
            # images[i] = images[i]/256
            images[i] = images[i]
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


def train_model():
    start_time = timeit.default_timer()
    # print(start_time) #19293.887149361

    training_data, test_data = loadMnistData()
    # train
    # classifier = svm.SVC()        #9443 trong  10000 gía trị đúng.
    classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False) #9998 trong 10000 gía trị đúng.

    # cho nó học từ images và label của data train
    classifier.fit(training_data[0], training_data[1])         
    train_time = timeit.default_timer()
    # print(train_time)       #19672.935809503
    print('gemfield train cost {}'.format(str(train_time - start_time) ) )

    # test
    print("Bắt đầu test!")
    pickle.dump(classifier, open("handwrite_model_kochia", 'wb'))

    #cho ra các label của test gọp lại thành mảng
    predictions = classifier.predict(test_data[0])
    predictions = []
    for a in classifier.predict(test_data[0]):
        predictions.append(a)
    print("PREDICT %r" % predictions)

    # so sánh cái mảng các label vừa được dự đoán được với mảng label mà ban đầu đã cho để xem có đúng thay hông??
    i = 0
    for a, y in zip(predictions, test_data[1]):
        if a == y:
            i = i + 1
    num_correct = i
    # print("predictions", predictions)  # [7,2,1,..]
    print("%s trong %s gía trị đúng." % (num_correct, len(test_data[1])))      

#     test_time = timeit.default_timer()
#     print('gemfield test cost {}'.format(str(test_time - train_time) ) )          #gemfield test cost 206.6903916629999

def test_model():
    classifier = pickle.load(open("handwrite_model", 'rb'))
    training_data, test_data = loadMnistData()
    result = classifier.score(test_data[0], test_data[1])
    print(result)

def predict_image(img):
    logo = img
    if type(img) is str:
        logo = io.imread(img)
    classifier = pickle.load(open("handwrite_model", 'rb'))
    show_image(logo)
    logo = logo.reshape(1, -1)
    result = classifier.predict(logo)
    print("RESULT %r" % result)
    return result

def show_image(img):
    logo = img.reshape(28, 28)
    print(logo.shape)
    for i in range(logo.shape[0]):
        for j in range(logo.shape[1]):
            if logo[i][j] > 0.0:
                print("@", end="");
            else:
                print("-", end="");
        print()


####-----------------------
train_model()
# test_model()
# training_dat,a test_data = loadMnistData()
# print(test_data[1][221:400])
# print(test_data[0][45])
# show_image(test_data[0][45])
predict_image(test_data[0][3775])
# predict_image("/home/teo/STUDY/images/image_90.jpg")
