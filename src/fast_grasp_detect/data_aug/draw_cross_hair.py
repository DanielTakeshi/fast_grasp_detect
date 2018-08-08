"""Don't delete this! We need it for bed-making.

Note that this used to be in a `visualizers` package, but all the other code
refers to it as `data_aug.draw_cross_hair`.
"""
import cv2
import cPickle as pickle
import IPython
import numpy as np

HALF_LENGTH = 40
RADIUS = 20
THICKNESS = 3
C_THICKNESS = 3
COLOR = (0,0,255) # RGB


class DrawPrediction:

	def make_cross_hair(self,image,p):
		cv2.circle(image,p,RADIUS,COLOR,C_THICKNESS)

		p1_h = (p[0] - HALF_LENGTH, p[1])
		p2_h = (p[0] + HALF_LENGTH, p[1])
		cv2.line(image,p1_h,p2_h,COLOR,THICKNESS)

		p1_v = (p[0], p[1] - HALF_LENGTH)
		p2_v = (p[0],p[1] + HALF_LENGTH)
		cv2.line(image,p1_v,p2_v,COLOR,THICKNESS)

		return image


	def draw_prediction(self,image,pose):
		x,y = pose
		pose = [int(x),int(y)]
		pose = tuple(pose)
		image = self.make_cross_hair(image,pose)
		return image


	def draw_tran_prediction(self,image,prediction):
		prob_success = int(prediction[0,0]*100)
		text_string = "P.S. "+str(prob_success)+'%'
		cv2.putText(image, text_string, (80,450), cv2.FONT_HERSHEY_SIMPLEX,4.0,(0,255,0),thickness=8)
		return image


if __name__ == "__main__":
	dp = DrawPrediction()
	path = cfg.ROLLOUT_PATH+'rollout_0/rollout.p'
	data = pickle.load(open(path,'rb'))
	grasp_point = data[0]
	box = grasp_point['label']['objects'][0]['box']
	x = int((box[0] + box[2])/2.0) 
	y = int((box[1]+ box[3])/2.0)
	pose = (x,y)
	c_img = grasp_point['c_img']
	image = dp.draw_prediction(c_img,pose)
	cv2.imshow('debug', image)
	cv2.waitKey(0)
	print "RESULT ", sc.check_success(wl)
