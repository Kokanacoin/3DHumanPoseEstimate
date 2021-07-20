import json
import numpy as np
import cv2 as cv
from tqdm import tqdm
import sys
import os
from Visualization import Visualization 
import random
import tensorflow as tf 
from PIL import Image  

'''
Train :	1877420		1420
Test  : 231151    1151
'''

def makeTrainY():
	subject_list = [1, 5, 6, 7, 8, 9]# not have 11
	action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	subaction_idx = (1, 2)
	count = 1 
	data = []
	new_data = []
	for each in subject_list:
	    with open('data/Human3.6M/annotations/Human36M_subject'+str(each)+'_joint_3d.json','r',encoding='utf8')as fp:
	        data.append(json.load(fp))
	for each_subject_data in range(len(data)):#每个大的文件
	    for i in data[each_subject_data].keys():            #i是2～16个
	        for j in data[each_subject_data][i].keys():     #j是 1或2
	            if subject_list[each_subject_data] == 11 and i == '2' and j == '2':
	                for t in range(1,4):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	            else:
	                for t in range(1,5):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	    print('now is ' + str(count))
	    count += 1
	a = np.array(new_data)
	np.save("data/Preprocess/train_y.npy", a)

def makeTestY():
	subject_list = [11]
	action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	subaction_idx = (1, 2)
	count = 1 
	data = []
	new_data = []
	for each in subject_list:
	    with open('data/Human3.6M/annotations/Human36M_subject'+str(each)+'_joint_3d.json','r',encoding='utf8')as fp:
	        data.append(json.load(fp))
	for each_subject_data in range(len(data)):#每个大的文件
	    for i in data[each_subject_data].keys():            #i是2～16个
	        for j in data[each_subject_data][i].keys():     #j是 1或2
	            if subject_list[each_subject_data] == 11 and i == '2' and j == '2':
	                for t in range(1,4):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	            else:
	                for t in range(1,5):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	    print('now is ' + str(count))
	    count += 1
	a = np.array(new_data)
	np.save("data/Preprocess/test_y.npy", a)

def makeTrainX():

	def make_Rectangele(x,y,w,h):

		if h > w:
			x_s = w // 2 + x - h // 2
			x_e = w // 2 + x + h // 2
			if x_s < 0: x_s = 0
			if x_s + h >= 999: x_s = 999 - h
			if y < 0: y = 0
			if y + h >= 1001: y = 1001 - h
			return x_s, y, h, h
		if h <= w:
			y_s = h // 2 + y - w // 2
			y_e = h // 2 + y + w // 2
			if x < 0: x = 0
			if x + w >= 999: x = 999 - w
			if y_s < 0: y_s = 0
			if y_s + w >= 1001: y_s = 1001 - w
			return x, y_s, w, w

	image_path = 'data/Human3.6M/images/'
	annotations_path = 'data/Human3.6M/annotations/'
	save_path = 'data/Preprocess/train_x/'

	subject_list = [1, 5, 6, 7, 8, 9]
	count = 0
	for sub in subject_list:
		with open(annotations_path + 'Human36M_subject'+str(sub)+'_data.json','r',encoding='utf8')as fp:
			data = json.load(fp)
		image_data = np.array(data['images'])
		annotations_data = np.array(data['annotations'])

		alls = len(image_data)

		
		for i in tqdm(range(alls)):
			try:
				x,y,width,height = [int(each) for each in annotations_data[i]['bbox']]
				src=cv.imread(image_path+image_data[i]['file_name'])       
				x,y,width,height = make_Rectangele(x,y,width,height)
				new_img = src[y:y+height,x:x+width]
				new_img = cv.resize(new_img, dsize=(128, 128))
				cv.imwrite(save_path + str(count) + '.jpg', new_img)
				count += 1
			except Exception as e:
				print('===========The error image id ' + str(i)+ '==========')
				print(image_data[i]['file_name'])
				print(e)
				sys.exit()
			finally:
				pass

def makeTestX():
	def make_Rectangele(x,y,w,h):
		if h > w:
			x_s = w // 2 + x - h // 2
			x_e = w // 2 + x + h // 2
			if x_s < 0: x_s = 0
			if x_s + h >= 999: x_s = 999 - h
			if y < 0: y = 0
			if y + h >= 1001: y = 1001 - h
			return x_s, y, h, h
		if h <= w:
			y_s = h // 2 + y - w // 2
			y_e = h // 2 + y + w // 2
			if x < 0: x = 0
			if x + w >= 999: x = 999 - w
			if y_s < 0: y_s = 0
			if y_s + w >= 1001: y_s = 1001 - w
			return x, y_s, w, w

	image_path = 'data/Human3.6M/images/'
	annotations_path = 'data/Human3.6M/annotations/'
	save_path = 'data/Preprocess/test_x/'

	subject_list = [11]
	count = 0
	for sub in subject_list:
		with open(annotations_path + 'Human36M_subject'+str(sub)+'_data.json','r',encoding='utf8')as fp:
			data = json.load(fp)
		image_data = np.array(data['images'])
		annotations_data = np.array(data['annotations'])

		alls = len(image_data)
		
		for i in tqdm(range(alls)):
			try:
				x,y,width,height = [int(each) for each in annotations_data[i]['bbox']]
				src=cv.imread(image_path+image_data[i]['file_name'])       
				x,y,width,height = make_Rectangele(x,y,width,height)
				new_img = src[y:y+height,x:x+width]
				new_img = cv.resize(new_img, dsize=(128, 128))
				cv.imwrite(save_path + str(count) + '.jpg', new_img)
				count += 1
			except Exception as e:
				print('===========The error image id ' + str(i)+ '==========')
				print(image_data[i]['file_name'])
				print(e)
				sys.exit()
			finally:
				pass

def walkFile():
	path = 'data/Preprocess/train_x/'
	count = 0
	for f in os.listdir(path):
		count += 1
	print("文件数量一共为:", count)

def make_tfrecord():
	cwd = os.getcwd()
	writer = tf.compat.v1.python_io.TFRecordWriter("train_own.tfrecords")

	joint = np.load("data/Preprocess/test_y.npy")

	path = 'data/Preprocess/test/0/'
	with tf.io.TFRecordWriter('train_own.tfrecords') as writer:
		for i in range(2000):
			image = Image.open(path  + str(i) +'.jpg')
			img_raw = image.tobytes()
			label = joint[i].reshape(-1).tobytes()
			feature = {                             # 建立 tf.train.Feature 字典
			    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),  
			    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))   
			}
			example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
			writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
		writer.close()

def cut_Test_y():
	subject_list = [11]
	action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	subaction_idx = (1, 2)
	count = 1 
	data = []
	new_data = []
	for each in subject_list:
	    with open('data/Human3.6M/annotations/Human36M_subject'+str(each)+'_joint_3d.json','r',encoding='utf8')as fp:
	        data.append(json.load(fp))
	for each_subject_data in range(len(data)):#每个大的文件
	    for i in data[each_subject_data].keys():            #i是2～16个
	        for j in data[each_subject_data][i].keys():     #j是 1或2
	            if subject_list[each_subject_data] == 11 and i == '2' and j == '2':
	                for t in range(1,4):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	            else:
	                for t in range(1,5):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	    print('now is ' + str(count))
	    count += 1

	a = np.array(new_data)
	p = []
	for i in range(len(a)):
		now_dir = str(i // 2000)
		now_id = str(i % 2000)
		p.append(a[i])

		if now_id == '1999' or i == len(a) - 1:
			os.mkdir('data/Preprocess/test/' + now_dir)
			p = np.array(p)
			np.save('data/Preprocess/test/' + now_dir +'/test_y.npy', p)
			p = []

def cut_Train_y():
	subject_list = [1, 5, 6, 7, 8, 9]
	action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	subaction_idx = (1, 2)
	count = 1 
	data = []
	new_data = []
	for each in subject_list:
	    with open('data/Human3.6M/annotations/Human36M_subject'+str(each)+'_joint_3d.json','r',encoding='utf8')as fp:
	        data.append(json.load(fp))
	for each_subject_data in range(len(data)):#每个大的文件
	    for i in data[each_subject_data].keys():            #i是2～16个
	        for j in data[each_subject_data][i].keys():     #j是 1或2
	            if subject_list[each_subject_data] == 11 and i == '2' and j == '2':
	                for t in range(1,4):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	            else:
	                for t in range(1,5):
	                    for e_image in data[each_subject_data][i][j].keys():
	                        new_data.append(data[each_subject_data][i][j][e_image])
	    print('now is ' + str(count))
	    count += 1

	a = np.array(new_data)
	p = []
	for i in range(len(a)):
		now_dir = str(i // 2000)
		now_id = str(i % 2000)
		p.append(a[i])

		if now_id == '1999' or i == len(a) - 1:
			os.mkdir('data/Preprocess/train/' + now_dir)
			p = np.array(p)
			np.save('data/Preprocess/train/' + now_dir +'/train_y.npy', p)
			p = []

def cut_Test_x():
	def make_Rectangele(x,y,w,h):
		if h > w:
			x_s = w // 2 + x - h // 2
			x_e = w // 2 + x + h // 2
			if x_s < 0: x_s = 0
			if x_s + h >= 999: x_s = 999 - h
			if y < 0: y = 0
			if y + h >= 1001: y = 1001 - h
			return x_s, y, h, h
		if h <= w:
			y_s = h // 2 + y - w // 2
			y_e = h // 2 + y + w // 2
			if x < 0: x = 0
			if x + w >= 999: x = 999 - w
			if y_s < 0: y_s = 0
			if y_s + w >= 1001: y_s = 1001 - w
			return x, y_s, w, w

	image_path = 'data/Human3.6M/images/'
	annotations_path = 'data/Human3.6M/annotations/'
	save_path = 'data/Preprocess/test/'

	subject_list = [11]
	count = 0
	for sub in subject_list:
		with open(annotations_path + 'Human36M_subject'+str(sub)+'_data.json','r',encoding='utf8')as fp:
			data = json.load(fp)
		image_data = np.array(data['images'])
		annotations_data = np.array(data['annotations'])

		alls = len(image_data)
		
		for i in tqdm(range(alls)):
			try:
				x,y,width,height = [int(each) for each in annotations_data[i]['bbox']]
				src=cv.imread(image_path+image_data[i]['file_name'])       
				x,y,width,height = make_Rectangele(x,y,width,height)
				new_img = src[y:y+height,x:x+width]
				new_img = cv.resize(new_img, dsize=(128, 128))

				dir_id = str(count // 2000)
				img_id = str(count % 2000)

				s_p = save_path + dir_id + '/'+ img_id+ '.jpg'
				cv.imwrite(s_p, new_img)
				count += 1
			except Exception as e:
				print('===========The error image id ' + str(i)+ '==========')
				print(image_data[i]['file_name'])
				print(e)
				sys.exit()
			finally:
				pass	

def cut_Train_x():
	def make_Rectangele(x,y,w,h):
		if h > w:
			x_s = w // 2 + x - h // 2
			x_e = w // 2 + x + h // 2
			if x_s < 0: x_s = 0
			if x_s + h >= 999: x_s = 999 - h
			if y < 0: y = 0
			if y + h >= 1001: y = 1001 - h
			return x_s, y, h, h
		if h <= w:
			y_s = h // 2 + y - w // 2
			y_e = h // 2 + y + w // 2
			if x < 0: x = 0
			if x + w >= 999: x = 999 - w
			if y_s < 0: y_s = 0
			if y_s + w >= 1001: y_s = 1001 - w
			return x, y_s, w, w

	image_path = 'data/Human3.6M/images/'
	annotations_path = 'data/Human3.6M/annotations/'
	save_path = 'data/Preprocess/train/'

	subject_list = [1, 5, 6, 7, 8, 9]
	count = 0
	for sub in subject_list:
		with open(annotations_path + 'Human36M_subject'+str(sub)+'_data.json','r',encoding='utf8')as fp:
			data = json.load(fp)
		image_data = np.array(data['images'])
		annotations_data = np.array(data['annotations'])

		alls = len(image_data)
		
		for i in tqdm(range(alls)):
			try:
				x,y,width,height = [int(each) for each in annotations_data[i]['bbox']]
				src=cv.imread(image_path+image_data[i]['file_name'])       
				x,y,width,height = make_Rectangele(x,y,width,height)
				new_img = src[y:y+height,x:x+width]
				new_img = cv.resize(new_img, dsize=(128, 128))

				dir_id = str(count // 2000)
				img_id = str(count % 2000)

				s_p = save_path + dir_id + '/'+ img_id+ '.jpg'
				cv.imwrite(s_p, new_img)
				count += 1
			except Exception as e:
				print('===========The error image id ' + str(i)+ '==========')
				print(image_data[i]['file_name'])
				print(e)
				sys.exit()
			finally:
				pass	


cut_Test_x()












