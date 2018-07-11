import os
import numpy as np
import IPython


class CONFIG(object):
    ###############PARAMETERS TO SWEEP##########

    def __init__(self):
        FIXED_LAYERS = 33
        #VARY {0, 4, 9}

        self.CONFIG_NAME = 'success_net'
        #self.ROOT_DIR    = '/media/autolab/1tb/daniel-bed-make/'
        self.ROOT_DIR    = '/nfs/diskstation/seita/bed-make/'
        self.NET_NAME    = '08_28_01_37_11save.ckpt-30300'
        #self.DATA_PATH   = self.ROOT_DIR+'bed_rcnn/'
        self.DATA_PATH   = self.ROOT_DIR+''

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
        self.CV_HELD_OUT_INDEX = 2
        self.PERFORM_CV = False

        # Various data paths. Note: BC_HELD_OUT is ignored if PERFORM_CV=True.
        self.ROLLOUT_PATH = self.DATA_PATH+'rollouts_nytimes/' # comment out if doing cross valid!!
        self.BC_HELD_OUT  = self.DATA_PATH+'held_out_nytimes/'
        self.IMAGE_PATH   = self.DATA_PATH+'images/'
        self.LABEL_PATH   = self.DATA_PATH+'labels/'
        self.CACHE_PATH   = self.DATA_PATH+'cache/'
        self.OUTPUT_DIR   = self.DATA_PATH+'output/'

        # Based on transitions
        self.OUTPUT_DIR        = self.DATA_PATH+'transition_output/'
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
        self.CLASSES = ['success','failure']
        # #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        #           'train', 'tvmonitor']
        self.NUM_LABELS = len(self.CLASSES)
        self.FLIPPED = False
        self.LIGHTING_NOISE = True
        self.QUICK_DEBUG = True

        # model parameter
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
        self.LEARNING_RATE = 0.1
        self.DECAY_STEPS = 30000
        self.DECAY_RATE = 0.1
        self.STAIRCASE = True
        self.BATCH_SIZE = 32
        self.MAX_ITER = 2000
        self.SUMMARY_ITER = 10
        self.TEST_ITER = 20
        self.SAVE_ITER = 500
        self.VIZ_DEBUG_ITER = 400
        self.CROSS_ENT_LOSS = False

        # test parameter
        self.PICK_THRESHOLD = 0.4
        self.THRESHOLD = 0.4
        self.IOU_THRESHOLD = 0.5

        # fast params
        self.FILTER_SIZE = 14
        self.NUM_FILTERS = 1024
        self.FILTER_SIZE_L1 = 7
        self.SIZE_L2 = 50176


    def compute_label(self,datum):
        """Interesting, Michael smoothed the label for his loss (L2^2, NOT the cross ent)."""
        clss = datum['class']
        label = np.zeros(2)
        if self.CROSS_ENT_LOSS:
            label[clss] = 1.0
            label[1-clss] = 0.0
        else:
            label[clss] = 0.99
        return label


    def get_empty_state(self, batchdim=None):
        """Each time we call a batch during training/testing, initialize with this."""
        if batchdim is not None:
            return np.zeros((batchdim, self.FILTER_SIZE, self.FILTER_SIZE, self.NUM_FILTERS))
        else:
            return np.zeros((self.BATCH_SIZE, self.FILTER_SIZE, self.FILTER_SIZE, self.NUM_FILTERS))


    def get_empty_label(self, batchdim=None):
        """Each time we call a batch during training/testing, initialize with this."""
        if batchdim is not None:
            return np.zeros((batchdim, 2))
        else:
            return np.zeros((self.BATCH_SIZE, 2))


    def break_up_rollouts(self,rollout):
        print("TODO: must check this method.")
        sys.exit()

        success_point = []
        success_rollout = []
        for data in rollout:
            if type(data) == list:
                continue
            if(data['type'] == 'success'):
                success_point.append(data)
            elif(data['type'] == 'grasp'):
                if( len(success_point) > 0):
                    success_rollout.append(success_point)
                    success_point = []
        success_rollout.append(success_point)
        return success_rollout
