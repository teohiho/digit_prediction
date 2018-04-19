import struct
import numpy as np
# import cv2

labelfile = open("train.csv")
magic, num = struct.unpack(">II", labelfile.read(8))
labelarray = np.fromstring(labelfile.read(), dtype=np.int8)

print(labelarray.shape)
print(labelarray[0:10])