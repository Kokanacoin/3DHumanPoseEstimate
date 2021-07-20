from AutoEncode import AutoEncode
from Visualization import Visualization
import cv2 as cv
import numpy as np

class DataSetForCNN:

	def __init__(self):
		self.dataSet_path = 'data/Preprocess/'			

	def extractPartImage(self,img):
	
		x = np.random.randint(0,15)
		y = np.random.randint(0,15)

		return img[x:x+112,y:y+112]

	def readData_FromAe2CNN(self):

		def makeTrainGenerator(auto):

			path = self.dataSet_path + 'train/'
			for file_name in range(938):
				for i in range(20):
					train_x = []
					for j in range(100):
						img = cv.imread(path + str(file_name) + '/' + str(i * 100 + j) + '.jpg')
						img = self.extractPartImage(img)
						train_x.append(img)
					train_y = auto.encoding(auto.train_x[file_name * 2000 + i * 100 :file_name * 2000 + (i + 1) * 100])
					train_x = np.array(train_x)
					train_x = train_x / 255
					yield ([train_x],[train_y])

		def makeTestGenerator(auto):
			path = self.dataSet_path + 'test/'
			for file_name in range(115):
				for i in range(20):
					test_x = []
					for j in range(100):
						img = cv.imread(path + str(file_name) + '/' + str(i * 100 + j) + '.jpg')
						img = self.extractPartImage(img)
						test_x.append(img)
					test_y = auto.encoding(auto.test_x[file_name * 2000 + i * 100 :file_name * 2000 + (i + 1) * 100])
					test_x = np.array(test_x)
					test_x = test_x / 255
					yield ([test_x],[test_y])
		auto = AutoEncode()
		auto.readData()
		auto.readModle('autoEncoder_v1.h5')
		trainGenerator = makeTrainGenerator(auto)
		testGenerator = makeTestGenerator(auto)

		return trainGenerator,testGenerator	

	