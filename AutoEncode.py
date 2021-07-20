from DataSet import DataSet
from Visualization import Visualization
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import os
import datetime

class AutoEncode:

	def __init__(self):

		os.environ['KMP_DUPLICATE_LIB_OK']='True'

		self.auto_encoder = None
		self.encoder_layer = None
		self.decoder_layer = None

		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None

		self.dataSet = DataSet()

	def makeEncodeNetwork(self,code_dim = 2000,regularizer = 0.1):
		input_layer = layers.Input(shape = (self.train_x.shape[1]),name = 'input_layer')
		code_layer = layers.Dense(code_dim,activation = 'relu',name = 'code_layer',kernel_regularizer=keras.regularizers.l2(regularizer))(input_layer)
		output_layer = layers.Dense(self.train_x.shape[1],activation = 'linear',name = 'output_layer')(code_layer)

		auto_encoder = keras.Model(input_layer,output_layer)
		auto_encoder.summary()
	
		encoder_layer = keras.Model(input_layer,code_layer)
	
		decoder_input = keras.Input((code_dim,))
		decoder_output = auto_encoder.layers[-1](decoder_input)

		decoder_layer = keras.Model(decoder_input, decoder_output)
		self.auto_encoder,self.encoder_layer,self.decoder_layer= auto_encoder,encoder_layer,decoder_layer

	def readData(self):

		self.train_x ,self.train_y ,self.test_x ,self.test_y = self.dataSet.readDate_ForAuto()

	def train(self,batch_size = 128,epochs = 10,alpha = 0.001):

		opt = tf.keras.optimizers.Adam(learning_rate=alpha)
		self.auto_encoder.compile(optimizer=opt,loss='mean_squared_error')
		self.auto_encoder.fit(self.train_x, self.train_y, batch_size=batch_size, epochs=epochs,validation_split=0.1)

	def saveModle(self,filename = None):
		path = 'output/'
		if filename is None:
			nowTime = datetime.datetime.now()
			self.auto_encoder.save(path + "autoEncoder_%s-%s-%s_%s_%s.h5"%(nowTime.year,nowTime.month,nowTime.day,nowTime.hour,nowTime.minute))
			
			# self.encoder_layer.save(path + 'encoder.h5')
			# self.decoder_layer.save(path + 'decoder.h5')


		elif filename is str:
			self.auto_encoder.save(path + filename)
		else:
			print('saveModle filename is wrong')

	def readModle(self,files):

		path = 'output/'

		self.auto_encoder = keras.models.load_model(path + files)

		en_input = layers.Input(shape = (51,))
		en_output = layers.Dense(2000,activation = 'relu',kernel_regularizer=keras.regularizers.l2(0.1))(en_input)

		self.encoder_layer = keras.Model(en_input,en_output)
		self.encoder_layer.set_weights(self.auto_encoder.layers[1].get_weights())


		de_input = layers.Input(shape = (2000,))
		de_output = layers.Dense(51,activation = 'linear')(de_input)

		self.decoder_layer = keras.Model(de_input,de_output)
		self.decoder_layer.set_weights(self.auto_encoder.layers[2].get_weights())


	def predict(self,predict_data_set = None):

		if self.auto_encoder == None: print('have not got model')
		else:
			if predict_data_set is None:
				return self.auto_encoder.predict(self.test_x)
			else:
				return self.auto_encoder.predict(predict_data_set)


	def encoding(self,data):
		if self.encoder_layer == None: print('have not got model')
		else:
			return self.encoder_layer.predict(data)

	
	def decoding(self,data):
		if self.decoder_layer == None: print('have not got model')
		else:
			return self.decoder_layer.predict(data)

if __name__ == '__main__':
	
	auto = AutoEncode()

	auto.readData()

	auto.makeEncodeNetwork()

	auto.train(epochs = 50)

	auto.saveModle()

	# auto.readModle("autoEncoder_v1.h5")


	# TEXT= 143930

	# vis = Visualization()

	# vis.show3D(auto.test_y[TEXT].reshape(17,3))

	# print(auto.test_x[TEXT].shape)

	# data = auto.predict(auto.test_x[TEXT:TEXT + 1])

	# print(data.shape)
	# vis.show3D(data.reshape(17,3))



 







