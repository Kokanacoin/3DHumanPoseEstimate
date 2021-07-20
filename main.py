from DataSet import DataSet
from Visualization import Visualization
from AutoEncode import AutoEncode
from CNN import CNN
import cv2 as cv
import numpy as np



vis = Visualization()
dataSet = DataSet()
cnn = CNN()


src=cv.imread('17.jpg')

img = cv.resize(src, dsize=(112, 112))

# img = dataSet.extractPartImage(src)

vis.show2D(img)

img = img[np.newaxis, :] / 255

print(img.dtype)

re = cnn.predict('final_v6_11408.h5',data = img)

print(re.shape)
vis.show3D(np.array(re[0]).reshape(17,3))

# vis = Visualization()
# auto = AutoEncode()
# auto.readData()
# auto.readModle('autoEncoder_v1.h5')

# data = auto.test_y[500:501].reshape(17,3)
# data_noice = auto.test_x[500:501].reshape(17,3)
# data_predict = np.array(auto.predict(auto.test_x[500:501])).reshape(17,3)



# vis.show3D(data)
# vis.show3D(data_noice)
# vis.show3D(data_predict)



