import os, sys
import numpy as np


class CONFIG(object):
    ###############PARAMETERS TO SWEEP##########

    def __init__(self):
        FIXED_LAYERS = 33
        #VARY {0, 4, 9}

        self.CONFIG_NAME = 'grasp_net'
        #self.ROOT_DIR    = '/media/autolab/1tb/daniel-bed-make/'   # Michael
        self.ROOT_DIR    = '/nfs/diskstation/seita/bed-make/'   # Tritons
        self.NET_NAME    = '08_28_01_37_11save.ckpt-30300'
        #self.DATA_PATH   = self.ROOT_DIR+'bed_rcnn/'   # Michael
        self.DATA_PATH   = self.ROOT_DIR+''     # Tritons

        # New, use for cross validation. Got this by randomly arranging numbers in a range.
        # Do this for my data, and comment out if otherwise.
        self.PERFORM_CV = False

        if self.PERFORM_CV:
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
            self.CV_HELD_OUT_INDEX = 2 # Adjust!
        else:
            # Now do this if I have a fixed held-out directory, as with Michael's data.
            # Note: BC_HELD_OUT is not used if PERFORM_CV=True.
            self.ROLLOUT_PATH = self.DATA_PATH+'rollouts_nytimes/'
            self.BC_HELD_OUT  = self.DATA_PATH+'held_out_nytimes/'

        # Other paths now.
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
        self.NUM_LABELS = len(self.CLASSES)
        self.FLIPPED = False # I don't think used?
        self.LIGHTING_NOISE = True # Is this used?
        self.QUICK_DEBUG = True # Is this used?

        # Model parameters. The USE_DEPTH is a critical one to test!
        self.T_IMAGE_SIZE_H = 480
        self.T_IMAGE_SIZE_W = 640
        self.IMAGE_SIZE = 448
        self.CELL_SIZE = 7
        self.BOXES_PER_CELL = 2
        self.ALPHA = 0.1
        self.DISP_CONSOLE = True
        self.RESOLUTION = 10
        self.USE_DEPTH = True

        # solver parameter
        self.FIX_PRETRAINED_LAYERS = True
        self.USE_EXP_MOV_AVG = False
        self.OPT_ALGO = 'ADAM'
        if self.OPT_ALGO == 'ADAM':
            self.LEARNING_RATE = 0.00010
        elif self.OPT_ALGO == 'SGD':
            self.LEARNING_RATE = 0.1
        else:
            raise ValueError(self.OPT_ALGO)
        self.DECAY_STEPS = 10000 # Decay every k steps
        self.DECAY_RATE = 0.1
        self.STAIRCASE = True
        self.BATCH_SIZE = 64
        self.MAX_ITER = 1000
        self.SUMMARY_ITER = 1
        self.TEST_ITER = 1
        self.SAVE_ITER = 100
        self.VIZ_DEBUG_ITER = 400

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
        """Labels scaled in [-1,1], as described in paper."""
        pose = datum['pose']
        label = np.zeros((2))
        x = pose[0]/self.T_IMAGE_SIZE_W - 0.5
        y = pose[1]/self.T_IMAGE_SIZE_H - 0.5
        label = np.array([x,y])
        return label


    def compare_preds_labels(self, preds, labels, doprint=False):
        """Try to get losses in the original pixel space for interpretability."""
        xx = np.array([self.T_IMAGE_SIZE_W, self.T_IMAGE_SIZE_H])
        raw_preds  = (0.5 + preds) * xx
        raw_labels = (0.5 + labels) * xx
        assert np.min(raw_labels) > 0.0
        delta = raw_preds - raw_labels
        test_loss_raw = np.linalg.norm(delta)
        if doprint:
            print("grasp test preds:\n{}".format(preds))
            print("grasp test labels:\n{}".format(labels))
            print("(raw) grasp test preds:\n{}".format(raw_preds))
            print("(raw) grasp test labels:\n{}".format(raw_labels))
        return test_loss_raw


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
        """I changed it to a way that makes more sense, a list of one-item lists."""

        #grasp_point = []
        #grasp_rollout = []
        #for data in rollout:
        #    if type(data) == list:
        #        continue
        #    if(data['type'] == 'grasp'):
        #        grasp_point.append(data)
        #    elif(data['type'] == 'success'):
        #        if( len(grasp_point) > 0):
        #            grasp_rollout.append(grasp_point)
        #            grasp_point = []
        #return grasp_rollout

        grasp_rollout = []
        for data in rollout:
            if type(data) is list:
                continue
            if data['type'] == 'grasp':
                grasp_rollout.append( [data] )
        return grasp_rollout
