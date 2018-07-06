import os
import numpy as np


class CONFIG(object):
    ###############PARAMETERS TO SWEEP##########

    def __init__(self):
        FIXED_LAYERS = 33

        #VARY {0, 4, 9}
        # SS_DATA = 0

        self.CONFIG_NAME = 'grasp_net'
        #self.ROOT_DIR    = '/media/autolab/1tb/daniel-bed-make/'   # Michael
        self.ROOT_DIR    = '/nfs/diskstation/seita/bed-make/'   # Tritons
        self.NET_NAME    = '08_28_01_37_11save.ckpt-30300'
        #self.DATA_PATH   = self.ROOT_DIR+'bed_rcnn/'   # Michael
        self.DATA_PATH   = self.ROOT_DIR+''     # Tritons

        # New, use for cross validation. Got this by randomly arranging numbers in a range.
        self.ROLLOUT_PATH = self.DATA_PATH+'rollouts/'
        self.CV_GROUPS = [
                [34,  7, 39, 37, 46],
                [16,  6,  8, 36, 26],
                [24, 11, 51, 38, 29],
                [32, 27,  9, 43, 19],
                [12, 35, 31,  4, 22],
                [13, 42,  5, 14, 25],
                [20, 40, 18, 21, 47],
                [23, 52, 28, 49, 45],
                [44, 48, 50, 15, 17],
                [ 3, 41, 10, 30, 33],
        ]
        self.CV_HELD_OUT_INDEX = 0
        self.PERFORM_CV = True

        # Various data paths. Note: BC_HELD_OUT is ignored if PERFORM_CV=True.
        self.BC_HELD_OUT  = self.DATA_PATH+'held_out_cal'
        self.IMAGE_PATH   = self.DATA_PATH+'images/'
        self.LABEL_PATH   = self.DATA_PATH+'labels/'
        self.CACHE_PATH   = self.DATA_PATH+'cache/'
        self.OUTPUT_DIR   = self.DATA_PATH+'output/'

        # Based on transitions
        self.TRAN_OUTPUT_DIR   = self.DATA_PATH+'transition_output/'
        self.TRAN_STATS_DIR    = self.TRAN_OUTPUT_DIR+'stats/'
        self.TRAIN_STATS_DIR_T = self.TRAN_OUTPUT_DIR+'train_stats/'
        self.TEST_STATS_DIR_T  = self.TRAN_OUTPUT_DIR+'test_stats/'

        # Based on grasping (NOTE: order matters, OUTPUT_DIR is updated)
        self.OUTPUT_DIR        = self.DATA_PATH+'grasp_output/'
        self.STAT_DIR          = self.OUTPUT_DIR+'stats/'
        self.TRAIN_STATS_DIR_G = self.OUTPUT_DIR+'train_stats/'
        self.TEST_STATS_DIR_G  = self.OUTPUT_DIR+'test_stats/'

        # Weights
        self.WEIGHTS_DIR = self.DATA_PATH+'weights/'
        #self.PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'
        self.PRE_TRAINED_DIR = '/nfs/diskstation/seita/yolo_tensorflow/data/pascal_voc/weights/'
        self.WEIGHTS_FILE = None
        # WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

        # Classes, labels, data augmentation
        self.CLASSES = ['success_grasp','fail_grasp']
        # #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        #           'train', 'tvmonitor']
        self.NUM_LABELS = len(self.CLASSES)
        self.FLIPPED = False
        self.LIGHTING_NOISE = True
        self.QUICK_DEBUG = True

        # model parameter

        #IMAGE_SIZE = 250
        self.T_IMAGE_SIZE_H = 480
        self.T_IMAGE_SIZE_W = 640
        self.IMAGE_SIZE = 448
        self.CELL_SIZE = 7
        self.BOXES_PER_CELL = 2
        self.ALPHA = 0.1
        self.DISP_CONSOLE = True
        self.RESOLUTION = 10
        self.USE_DEPTH = False

        # solver parameter

        self.GPU = ''
        self.LEARNING_RATE = 0.1
        self.DECAY_STEPS = 30000
        self.DECAY_RATE = 0.1
        self.STAIRCASE = True
        self.BATCH_SIZE = 45
        self.MAX_ITER = 1000#30000
        self.SUMMARY_ITER = 10
        self.TEST_ITER = 20
        self.SAVE_ITER = 500
        self.VIZ_DEBUG_ITER = 400

        # test parameter

        #THRESHOLD = 0.0008
        self.PICK_THRESHOLD = 0.4
        self.THRESHOLD = 0.4
        self.IOU_THRESHOLD = 0.5
        #IOU_THRESHOLD = 0.0001

        # fast params
        self.FILTER_SIZE = 14
        self.NUM_FILTERS = 1024
        self.FILTER_SIZE_L1 = 7
        self.SIZE_L2 = 50176


    def compute_label(self,datum):
        """Labels scaled in [-1,1], as described in paper."""
        pose = datum['pose']
        label = np.zeros((2))
        x = pose[0]/self.T_IMAGE_SIZE_W-0.5
        y = pose[1]/self.T_IMAGE_SIZE_H-0.5
        label = np.array([x,y])
        return label


    def get_empty_state(self):
        return np.zeros((self.BATCH_SIZE, self.FILTER_SIZE, self.FILTER_SIZE, self.NUM_FILTERS))


    def get_empty_label(self):
        return np.zeros((self.BATCH_SIZE, 2))


    def break_up_rollouts(self,rollout):
        grasp_point = []
        grasp_rollout = []
        for data in rollout:
            if type(data) == list:
                continue
            if(data['type'] == 'grasp'):
                grasp_point.append(data)
            elif(data['type'] == 'success'):
                if( len(grasp_point) > 0):
                    grasp_rollout.append(grasp_point)
                    grasp_point = []
        return grasp_rollout
