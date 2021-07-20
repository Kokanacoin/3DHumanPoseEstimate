import cv2 as cv
import random
import numpy as np
from PIL import Image
from Visualization import Visualization

class DataSet():
	
	def __init__(self):

		self.dataSet_path = 'data/Preprocess/'				
		self.joint_num = 17
		self.subject_list = [1, 5, 6, 7, 8, 9, 11]
		self.action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
		self.subaction_idx = (1, 2)
		self.camera_idx = (1, 2, 3, 4)
		self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
	
	def makeImageFolder(self,subject_id,act_id,subact_id,camera_id):

		return "s_%02d_act_%02d_subact_%02d_ca_%02d" % (subject_id, act_id, subact_id, camera_id)

	def makeGauss(self,lists,mean = 0,var = 40.):
		noice = np.random.normal(loc = mean, scale = var, size=lists.shape)
		return lists + noice

	def readDate_ForAuto(self):
		
		train_y = np.load(self.dataSet_path + 'train_y.npy')
		test_y = np.load(self.dataSet_path + 'test_y.npy')
		
		train_y = train_y.reshape(train_y.shape[0],-1)
		test_y = test_y.reshape(test_y.shape[0],-1)
		
		train_x = self.makeGauss(train_y)
		test_x = self.makeGauss(test_y)

		return train_x, train_y, test_x, test_y

	def extractPartImage(self,img):
	
		x = np.random.randint(0,15)
		y = np.random.randint(0,15)

		return img[x:x+112,y:y+112]

	def readDate_ForCNN_ALL(self):
		def makeTrainGenerator():
			path = self.dataSet_path + 'train/'
			for file_name in range(938):
				for i in range(20):
					train_x = []
					train_y = np.load(path + str(file_name) + '/' + 'train_y.npy')
					train_y = train_y.reshape(train_y.shape[0],-1)
					for j in range(100):
						img = cv.imread(path + str(file_name) + '/' + str(i * 100 + j) + '.jpg')
						img = self.extractPartImage(img)
						train_x.append(img)
		
					train_x = np.array(train_x)
					train_x = train_x / 255
					yield ([train_x],[train_y[i*100:(i + 1) * 100]])
		def makeTestGenerator():
			path = self.dataSet_path + 'test/'
			for file_name in range(115):
				for i in range(20):
					test_x = []
					test_y = np.load(path + str(file_name) + '/' + 'test_y.npy')
					test_y = test_y.reshape(test_y.shape[0],-1)
					for j in range(100):
						img = cv.imread(path + str(file_name) + '/' + str(i * 100 + j) + '.jpg')
						img = self.extractPartImage(img)
						test_x.append(img)
		
					test_x = np.array(test_x)
					test_x = test_x / 255
					yield ([test_x],[test_y[i*100:(i + 1) * 100]])

		trainGenerator = makeTrainGenerator()
		testGenerator = makeTestGenerator()

		return trainGenerator,testGenerator

	def readDate_FromAe2CNN(self):

		def makeTrainGenerator(auto):

			path = self.dataSet_path + 'train/'
			for file_name in range(938):
				train_x = []
				for i in range(2000):
					img = cv.imread(path + str(file_name) + '/' + str(i) + '.jpg')
					img = self.extractPartImage(img)
					train_x.append(img)
				train_y = auto.encoding(auto.train_x[file_name * 2000:(file_name+1) * 2000])
				yield (train_x,train_y)

		def makeTestGenerator(auto):
			path = self.dataSet_path + 'test/'
			for file_name in range(115):
				test_x = []
				for i in range(2000):
					img = cv.imread(path + str(file_name) + '/' + str(i) + '.jpg')
					img = self.extractPartImage(img)
					test_x.append(img)
				test_y = auto.encoding(auto.test_x[file_name * 2000:(file_name+1) * 2000])
				yield (test_x,test_y)
		auto = AutoEncode()
		auto.readDate()
		auto.readModel('autoEncoder_v1.h5')
		trainGenerator = makeTrainGenerator(auto)
		testGenerator = makeTestGenerator(auto)

		return trainGenerator,testGenerator	

if __name__ == '__main__':
	d = DateSet()
	train,test = d.readDate_FromAe2CNN()

	for each in test:
		print(each)
		break
	








