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
    i = 0;
    for img_file,label_file,items in zip(['train-images-idx3-ubyte','t10k-images-idx3-ubyte'],
                                   ['train-labels-idx1-ubyte','t10k-labels-idx1-ubyte'],
                                   [TRAIN_ITEMS, TEST_ITEMS]):

#-------------
        # print("img_file",img_file)
        # print("label_file",label_file)
        # print("items",items)
        ## img_file train-images-idx3-ubyte
        ## label_file train-labels-idx1-ubyte
        ## items 60000
        ## img_file t10k-images-idx3-ubyte
        ## label_file t10k-labels-idx1-ubyte
        ## items 10000
#-------------
        data_img = open(img_file, 'rb').read()
        # print("data_img:",data_img)
        ## b'\x01\x00\x02\x00\x05\x00\x00\x00\xbd\x01\x00\x00\x00\x00\x00\x00'
#-------------
        data_label = open(label_file, 'rb').read()
        ## i la số nguyên, iiii là 4 số nguyên
        fmt = '>iiii'   #format
        # print("fmt:", fmt)  #fmt: >iiii

        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset) #giai nen du lieu
        # print("magic_number: ", magic_number)
        # print("img_number: ", img_number)
        # print("height: ", height)
        # print("width: ", width)
        # # magic_number:  2051
        # # img_number:  60000
        # # height:  28
        # # width:  28
        # # magic_number:  2051
        # # img_number:  10000
        # # height:  28
        # # width:  28
#-------------
        # print('magic number is {}, image number is {}, height is {} and width is {}'.format(magic_number, img_number, height, width))
        # # magic number is 2051, image number is 60000, height is 28 and width is 28
        # # magic number is 2051, image number is 10000, height is 28 and width is 28
#-------------
        # #slide over the 2 numbers above
        offset += struct.calcsize(fmt) # trả lại kích thước của cấu trúc
        # print("offset: ", offset)       #offset:  16

        # #28x28
        image_size = height * width
        # #B means unsigned char
        fmt = '>{}B'.format(image_size)
        # print("fmt:", fmt)      #fmt: >784B

        # #because gemfield has insufficient memory resource
        if items > img_number:
            items = img_number
            print("có không??")
        images = np.empty((items, image_size))
        # print("images.size:" , images.size)
        # # images.size: 47040000 (60000x784)
        # # images.size: 7840000
        # print("images:" , images)
        # # images: [[0. 0. 0. ... 0. 0. 0.]
        # #          [0. 0. 0. ... 0. 0. 0.]
        # #          [0. 0. 0. ... 0. 0. 0.]
        # #          ...
        # #          [0. 0. 0. ... 0. 0. 0.]
        # #          [0. 0. 0. ... 0. 0. 0.]
        # #          [0. 0. 0. ... 0. 0. 0.]]
        # print("images[0]:" , images[0].size) #784

#-------------

        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
            # print("images[{}]:{}", i, images[i])
            # # images[{}]:{} 9993 [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
            # #                    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
            # #                    ....
            # #                    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
            # #                    0.   0.   0. 110. 109. 109.  47.   0.   0.   0.   0.   0.   0.   0.
            #0~255 to 0~1

            # if i== 0:
            #     plt.imshow(images[0])
            #     plt.show()

            images[i] = images[i]/256
            # print("images[0]: ", images[0]/256)
            # # images[0]:  [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
            # #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
            # #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
            # #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
            # #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00


            offset += struct.calcsize(fmt)
            # print("offset: ",offset) #7840016



        # #fmt of struct unpack, > means big endian, i means integer, well, ii mean 2 integers
        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)

        # print('magic number is {} and label number is {}'.format(magic_number, label_number))
        # # magic number is 2049 and label number is 60000
        # # magic number is 2049 and label number is 10000


        # #slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        # print("offset:" , offset) #offset: 8

        # #B means unsigned char
        fmt = '>B'
        # print("fmt: ", fmt)         #fmt:  >B

        # #because gemfield has insufficient memory resource
        if items > label_number:
            items = label_number
            print("có khong??")
        labels = np.empty(items) 
        # print("labels:", labels)        #labels: [0. 0. 0. ... 0. 0. 0.]
        # print("items: ", items)
        # # items:  60000
        # # items:  10000
        # print("labels[0]:" , labels[0]) #labels[0]: 0.0

        for i in range(2):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)
            # print(labels[i]) #khi range(2): 5,0,7,2
        
        mnist_data.append((images, labels.astype(int)))
    return mnist_data

# print("loadMnistData: ", loadMnistData())
# # loadMnistData:  [(array([[0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        ...,
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.]]), array([5, 0, 0, ..., 0, 0, 0])), (array([[0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        ...,
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.],
# #        [0., 0., 0., ..., 0., 0., 0.]]), array([7, 2, 0, ..., 0, 0, 0]))]



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
    pickle.dump(classifier, open("handwrite_model", 'wb'))

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
    if type(img) == 'string':
        logo = io.imread(img)
    logo = img
    classifier = pickle.load(open("handwrite_model", 'rb'))
    logo = logo.reshape(1, -1)
    # show_image(img)
    result = classifier.predict(logo)
    print("RESULT %r" % result)
    return result

def show_image(img):
    logo = io.imread(img)
    for i in range(logo.shape[0]):
        for j in range(logo.shape[1]):
            print("%3d" % logo[i][j], end="");
        print()


####-----------------------
train_model()
# test_model()
# training_data, test_data = loadMnistData()
# print(test_data[1][221:400])
# predict_image(test_data[0][9])

