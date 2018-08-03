import os
import numpy as np


class CONFIG(object):

    def __init__(self, args):
        """Some manual work needed for CV to put groupings here.

        But external code can loop through the CV indices.
        To find 10-fold CV for a group of N rollouts, do:

            [list(x) for x in np.array_split( np.random.permutation(N) , 10) ]

        and paste the result in `CV_GROUPS`.
        """
        self.args = args
        self.PERFORM_CV = args.do_cv
        self.PRINT_PREDS = args.print_preds

        self.CONFIG_NAME = 'success_net'
        self.ROOT_DIR    = '/nfs/diskstation/seita/bed-make/'   # Tritons
        self.DATA_PATH   = self.ROOT_DIR+''                     # Tritons
        self.NET_NAME    = '08_28_01_37_11save.ckpt-30300'

        if self.PERFORM_CV:
            assert args.cv_idx is not None
            self.ROLLOUT_PATH = self.DATA_PATH+'rollouts_white_v01/'
            self.CV_GROUPS = [
                [10, 4, 27, 3, 23],
                [34, 26, 35, 45, 28],
                [12, 7, 11, 5, 47],
                [46, 17, 15, 9, 20],
                [6, 49, 16, 18, 24],
                [30, 41, 29, 44, 48],
                [8, 33, 32, 31, 14],
                [2, 38, 42, 21, 19],
                [25, 37, 0, 39, 36],
                [43, 1, 40, 22, 13]
            ]
            self.CV_HELD_OUT_INDEX = args.cv_idx
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

        # If training success net, info goes here. Order matters, OUTPUT_DIR is updated.
        self.OUTPUT_DIR        = self.DATA_PATH+'transition_output/'
        self.STAT_DIR          = self.OUTPUT_DIR+'stats/'
        self.TRAIN_STATS_DIR_G = self.OUTPUT_DIR+'train_stats/'
        self.TEST_STATS_DIR_G  = self.OUTPUT_DIR+'test_stats/'

        # Weights
        self.WEIGHTS_DIR = self.DATA_PATH+'weights/'
        self.PRE_TRAINED_DIR = '/nfs/diskstation/seita/yolo_tensorflow/data/pascal_voc/weights/'
        self.WEIGHTS_FILE = None
        # WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

        # Classes, labels, data augmentation
        self.CLASSES = ['success','failure']
        self.NUM_LABELS = len(self.CLASSES)
        ## self.FLIPPED = False
        ## self.LIGHTING_NOISE = True
        ## self.QUICK_DEBUG = True

        # Model parameters. The USE_DEPTH is a critical one to test!
        self.T_IMAGE_SIZE_H = 480
        self.T_IMAGE_SIZE_W = 640
        self.IMAGE_SIZE = 448
        self.CELL_SIZE = 7
        self.BOXES_PER_CELL = 2
        self.ALPHA = 0.1
        self.DISP_CONSOLE = True
        self.RESOLUTION = 10
        self.USE_DEPTH = False # False means RGB

        # solver parameter
        # Careful, if fixing, this means our images effectively turn into (fs,fs,channels),
        # e.g., could be (14,14,1024). If False, we train from the usual (480,640,3) images.
        self.FIX_PRETRAINED_LAYERS = True
        self.OPT_ALGO = 'ADAM'
        if self.OPT_ALGO == 'ADAM':
            self.LEARNING_RATE = 0.00010
            self.USE_EXP_MOV_AVG = False
        elif self.OPT_ALGO == 'SGD':
            self.LEARNING_RATE = 0.01
            self.USE_EXP_MOV_AVG = True
        else:
            raise ValueError(self.OPT_ALGO)
        self.DECAY_STEPS = 10000 # Decay every k steps
        self.DECAY_RATE = 0.1
        self.STAIRCASE = True
        self.BATCH_SIZE = 64
        self.MAX_ITER = args.max_iters
        self.SUMMARY_ITER = 1
        self.TEST_ITER = 1
        self.SAVE_ITER = 100
        self.VIZ_DEBUG_ITER = 400 # ?
        self.CROSS_ENT_LOSS = True # Michael did 'softened L2' earlier

        ## # test parameter
        ## self.PICK_THRESHOLD = 0.4
        ## self.THRESHOLD = 0.4
        ## self.IOU_THRESHOLD = 0.5

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


    def compare_preds_labels(self, preds, labels, correctness, doprint=False):
        """For success, we have correctness to consider instead."""
        if doprint:
            print("success test preds:\n{}".format(preds))
            print("success test labels:\n{}".format(labels))
        print("correctness:\n{}".format(correctness))


    def get_empty_state(self, batchdim=None):
        """Each time we call a batch during training/testing, initialize with this."""
        bs = self.BATCH_SIZE
        if batchdim is not None:
            bs = batchdim

        if self.FIX_PRETRAINED_LAYERS:
            return np.zeros((bs, self.FILTER_SIZE, self.FILTER_SIZE, self.NUM_FILTERS))
        else:
            assert self.IMAGE_SIZE == 448
            return np.zeros((bs, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))


    def get_empty_label(self, batchdim=None):
        """Each time we call a batch during training/testing, initialize with this."""
        if batchdim is not None:
            return np.zeros((batchdim, 2))
        else:
            return np.zeros((self.BATCH_SIZE, 2))


    def break_up_rollouts(self,rollout):
        """I changed it to a way that makes more sense, a list of one-item lists."""
        success_rollout = []
        for data in rollout:
            if type(data) is list:
                continue
            if data['type'] == 'success':
                success_rollout.append( [data] )
        return success_rollout
