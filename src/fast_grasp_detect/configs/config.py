"""
Do not delete this file. We use it for the labeler later for data collection.
"""

import os

#
# path and dataset parameter
#


class CONFIG(object):
###############PARAMETERS TO SWEEP##########


	def __init__(self):
		FIXED_LAYERS = 33

		#VARY {0, 4, 9}
		# SS_DATA = 0
		self.CONFIG_NAME = 'SS_0'

		self.ROOT_DIR = '/media/autolab/1tb/daniel-bed-make/'

		self.NET_NAME = '08_28_01_37_11save.ckpt-30300'
		self.DATA_PATH = self.ROOT_DIR + 'bed_rcnn/'

		# ROLLOUT_PATH = DATA_PATH+'rollouts/'
		# BC_HELD_OUT = DATA_PATH+'held_out_bc'

		self.ROLLOUT_PATH = self.DATA_PATH+'rollouts_dart/'
		self.BC_HELD_OUT = self.DATA_PATH+'held_out_dart'



		self.IMAGE_PATH = self.DATA_PATH+'images/'
		self.LABEL_PATH = self.DATA_PATH+'labels/'

		self.CACHE_PATH = self.DATA_PATH+'cache/'

		self.OUTPUT_DIR = self.DATA_PATH +'output/'

		self.TRAN_OUTPUT_DIR = self.DATA_PATH +'transition_output/' 
		self.TRAN_STATS_DIR = self.TRAN_OUTPUT_DIR + 'stats/'
		self.TRAIN_STATS_DIR_T = self.TRAN_OUTPUT_DIR + 'train_stats/'
		self.TEST_STATS_DIR_T = self.TRAN_OUTPUT_DIR + 'test_stats/'


		self.GRASP_OUTPUT_DIR = self.DATA_PATH + 'grasp_output/'
		self.GRASP_STAT_DIR = self.GRASP_OUTPUT_DIR + 'stats/' 
		self.TRAIN_STATS_DIR_G = self.GRASP_OUTPUT_DIR + 'train_stats/'
		self.TEST_STATS_DIR_G = self.GRASP_OUTPUT_DIR + 'test_stats/'

		self.WEIGHTS_DIR = self.DATA_PATH + 'weights/'

		self.PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'

		#ROLLOUT_PATH = DATA_PATH+'rollouts/'

		self.WEIGHTS_FILE = None


		# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

		self.CLASSES = ['success','failure']

		# #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
		#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
		#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
		# 			'train', 'tvmonitor']

		self.NUM_LABELS = len(self.CLASSES)

		self.FLIPPED = False
		self.LIGHTING_NOISE = True


		self.QUICK_DEBUG = True

		#
		# model parameter
		#

		#IMAGE_SIZE = 250
		self.T_IMAGE_SIZE_H = 480
		self.T_IMAGE_SIZE_W = 640

		self.IMAGE_SIZE = 448
		self.CELL_SIZE = 7

		self.BOXES_PER_CELL = 2

		self.ALPHA = 0.1

		self.DISP_CONSOLE = True



		self.RESOLUTION = 10


		#
		# solver parameter
		#

		self.GPU = ''

		self.LEARNING_RATE = 0.1
		


		self.DECAY_STEPS = 30000

		self.DECAY_RATE = 0.1

		self.STAIRCASE = True

		self.BATCH_SIZE = 45

		#MAX_ITER = 200
		self.MAX_ITER = 1000#30000

		self.SUMMARY_ITER = 10
		self.TEST_ITER = 20
		self.SAVE_ITER = 500

		self.VIZ_DEBUG_ITER = 400
		#
		# test parameter
		#

		#THRESHOLD = 0.0008
		self.PICK_THRESHOLD = 0.4
		self.THRESHOLD = 0.4
		self.IOU_THRESHOLD = 0.5
		#IOU_THRESHOLD = 0.0001

		#FAST PARAMS
		self.FILTER_SIZE = 14
		self.NUM_FILTERS = 1024

		self.FILTER_SIZE_L1 = 7

		self.SIZE_L2 = 50176
