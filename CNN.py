from DataSet import DataSet
from DataSetForCNN import DataSetForCNN
import os 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import tensorflow.keras as keras
from Visualization import Visualization
from AutoEncode import AutoEncode


class CNN:

	def __init__(self):
		
		os.environ['KMP_DUPLICATE_LIB_OK']='True'
		self.modelCNN = None
		self.dataSetForCNN = DataSetForCNN()
		self.dataSet = DataSet()
		self.final_CNN = None


	def makeModel(self):

		model = Sequential()

		model.add(Conv2D(input_shape=(112,112,3),filters=36, kernel_size=(9,9), padding='VALID',activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(Conv2D(filters=72, kernel_size=(5, 5), padding='VALID', activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(Conv2D(filters=72, kernel_size=(5, 5), padding='VALID', activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(BatchNormalization())

		model.add(Flatten())

		model.add(Dense(512, activation='relu'))  
		model.add(Dropout(0.5)) 

		model.add(Dense(2048, activation='relu'))  
		model.add(Dropout(0.5)) 

		model.add(Dense(4096, activation='relu'))  
		model.add(Dropout(0.5))

		model.add(Dense(2000, activation='linear')) 

		model.summary()

		self.modelCNN = model


	def saveModel(self,param):
		self.modelCNN.save('output/CNN_epoch'+str(param + 1)+'.h5')

	def train(self,epoch):


		opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
		self.modelCNN.compile(optimizer=opt,loss='mean_squared_error')

		for i in range(epoch):
			trainGenerator,testGenerator = self.dataSetForCNN.readData_FromAe2CNN()
			self.modelCNN.fit_generator(trainGenerator,steps_per_epoch=18760)

			if i % 5 == 0:
				self.saveModel(i)

	def splicingNet(self):
		path = 'output/'

		model_ed = keras.models.load_model(path + 'CNN_epoch46.h5')
		auto_ed = keras.models.load_model(path + 'autoEncoder_v1.h5')


		# print(len(model_ed.layers[0].get_weights()[0]))
		# print(len(model_ed.layers[0].get_weights()[1]))



		model = Sequential()

		model.add(Conv2D(input_shape=(112,112,3),filters=36, kernel_size=(9,9), padding='VALID',activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(Conv2D(filters=72, kernel_size=(5, 5), padding='VALID', activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(Conv2D(filters=72, kernel_size=(5, 5), padding='VALID', activation='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(BatchNormalization())
		model.add(Flatten())

		model.add(Dense(512, activation='relu'))  
		model.add(Dropout(0.5)) 

		model.add(Dense(2048, activation='relu'))  
		model.add(Dropout(0.5)) 

		model.add(Dense(4096, activation='relu'))  
		model.add(Dropout(0.5))

		model.add(Dense(2000, activation='linear')) 

		model.add(Dense(51, activation='linear')) 
		model.summary()

		for i in range(15):
			model.layers[i].set_weights(model_ed.layers[i].get_weights())


		model.layers[-1].set_weights(auto_ed.layers[-1].get_weights())


		for i in range(4):
			model.layers[i].trainable = False
		self.final_CNN = model

	def sava_final(self,param):
		self.final_CNN.save('output/final_epoch'+str(param + 1)+'.h5')


	def train_final(self,epoch):

		opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.final_CNN.compile(optimizer=opt,loss='mean_squared_error')
		for i in range(epoch):
			trainGenerator,testGenerator = self.dataSet.readDate_ForCNN_ALL()
			self.final_CNN.fit_generator(trainGenerator,steps_per_epoch=18760)
			if (i + 1 ) % 3 == 0:
				self.sava_final(i)

	def predict(self,model,data =None):
		path = 'output/'
		model = keras.models.load_model(path + model)

		if data is  None:
			trainGenerator,testGenerator = self.dataSet.readDate_ForCNN_ALL()
			data = testGenerator.__next__()
			data = data[0][0]	
		return model.predict(data)

	def train_continue(self,epoch):


		path = 'output/'
		self.final_CNN = keras.models.load_model(path + 'final_epoch24.h5')


		opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.final_CNN.compile(optimizer=opt,loss='mean_squared_error')
		for i in range(epoch):
			trainGenerator,testGenerator = self.dataSet.readDate_ForCNN_ALL()
			self.final_CNN.fit_generator(trainGenerator,steps_per_epoch=18760)
			if (i + 1 ) % 5 == 0:
				self.sava_final(i)


if __name__ == '__main__':
	c = CNN()
	c.train_continue(25)
	











