import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json

class Visualization():

	def __init__(self):
		self.images_path = 'data/Human3.6M/images/'				#图片路径
		self.annotations_path = 'data/Human3.6M/annotations/'	#姿势标志
		self.joint_num = 17
		self.subject_list = [1, 5, 6, 7, 8, 9, 11]
		self.action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
		self.subaction_idx = (1, 2)
		self.camera_idx = (1, 2, 3, 4)
		self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
		self.human36m_connectivity_dict = [
											[10,9,0],
											[9,8,0],
											[8,14,0],
											[14,15,0],
											[15,16,0],
											[8,11,1],
											[11,12,1],
											[12,13,1],
											[8,7,0],
											[7,0,0],
											[0,1,0],
											[1,2,0],
											[2,3,0],
											[0,4,1],
											[4,5,1],
											[5,6,1]]

	def getJointID(self,image_name):
		
		arr = np.load("count.npy")
		s = self.subject_list.index(int(image_name[2:4]))
		act = self.action_idx.index(int(image_name[9:11]))
		subact = self.subaction_idx.index(int(image_name[19:21]))
		ca = self.camera_idx.index(int(image_name[25:27]))
		image_id = int(image_name[28:34]) - 1
		id_start = arr[s * (15*2*4) + act * (2 * 4)][4] 
		all_id = arr[s * (15*2*4) + act * (2 * 4) + subact * (4) + ca][4] + image_id
		return all_id - id_start

	def getImageID(self,image_name):

		arr = np.load("count.npy")
		s = self.subject_list.index(int(image_name[2:4]))
		act = self.action_idx.index(int(image_name[9:11]))
		subact = self.subaction_idx.index(int(image_name[19:21]))
		ca = self.camera_idx.index(int(image_name[25:27]))
		image_id = int(image_name[28:34]) - 1
		all_id = arr[s * (15*2*4) + act * (2 * 4) + subact * (4) + ca][4] + image_id
		this_subject_id = all_id - arr[s * (15*2*4)][4]
		return all_id,this_subject_id

	def draw3Dpose(self,pose_3d, ax, show_style,lcolor="#3498db", rcolor="#e74c3c",add_labels=False):  
	    	
		if show_style is 'line':
			for i in self.human36m_connectivity_dict:
				x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
				ax.plot(x, y, z, lw=2, c=lcolor if i[2] else rcolor)
		elif show_style is 'point':
			x = [x[0] for x in pose_3d]
			y = [x[1] for x in pose_3d]
			z = [x[2] for x in pose_3d]
			ax.scatter(x, y, z, c = 'r') 
		RADIUS = 750 
		xroot, yroot, zroot = pose_3d[5, 0], pose_3d[5, 1], pose_3d[5, 2]
		ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
		ax.set_zlim3d([0, 2 * RADIUS + zroot])
		ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

	def show3DPoseDataSet(self,path,show_style,json_file = None):
		
		path = path.split('/')[1]
		if json_file is None:
			json_file_name = 'data/Human3.6M/annotations/Human36M_subject' + str(int(path[2:4]))+ '_joint_3d.json'
			
			with open(json_file_name,'r',encoding='utf8')as fp:
				data = json.load(fp)
		else:
			data = json_file
		ids = self.getJointID(path)
		joint_data = data[str(int(path[9:11]))][str(int(path[19:21]))][str(int(path[28:34]) - 1)]
		joint_data = np.array(joint_data)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		self.draw3Dpose(joint_data, ax,show_style)
		plt.show()

	def show2DOriginalPhoto(self,path):

		if type(path) is str:
			path = self.images_path + path
			src=cv.imread(path) 
			cv.namedWindow('origina_photo', cv.WINDOW_AUTOSIZE)
			cv.imshow('origina_photo', src)
			cv.waitKey(0)
			cv.destroyAllWindows()
		else :
			src = path
			cv.namedWindow('origina_photo', cv.WINDOW_AUTOSIZE)
			cv.imshow('origina_photo', src)
			cv.waitKey(0)
			cv.destroyAllWindows()
		
	def show2DFoundPhoto(self,path,show_style,json_file = None):

		'''
		show_style@param:
		human_only
		square_only
		square
		'''
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

		if json_file is None:
			json_file_name = 'data/Human3.6M/annotations/Human36M_subject' + str(int(path[2:4]))+ '_data.json'
			with open(json_file_name,'r',encoding='utf8')as fp:
				data = json.load(fp)
		else:
			data = json_file
		image_data = np.array(data['images'])
		annotations_data = np.array(data['annotations'])
		image_name = path.split('/')[1]
		all_id, image_id = self.getImageID(image_name)		
		x,y,width,height = [int(each) for each in annotations_data[image_id]['bbox']]
		src=cv.imread(self.images_path+image_data[image_id]['file_name'])       
		cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)

		if show_style is 'human_only':pass
		if show_style is 'square' or show_style is 'square_only': 
			x,y,width,height = make_Rectangele(x,y,width,height)

		if show_style is 'square_only':
			new_img = src[y:y+height,x:x+width]
			new_img = cv.resize(new_img, dsize=(128, 128))
			print('Image shape : ' + str(new_img.shape))
			cv.imshow('input_image', new_img)

		else:
			cv.rectangle(src,(x,y),(x + width,y + height),(0,255,0),3)
			cv.imshow('input_image', src)
		cv.waitKey(0)
		cv.destroyAllWindows()

	def show2D(self,img):
		
		cv.imshow('image', img)
		cv.waitKey(0)
		cv.destroyAllWindows()

	def show3D(self,joint_data):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		self.draw3Dpose(joint_data, ax,'line')
		plt.show()
















