import os, sys
from os.path import join
import numpy as np


class CONFIG(object):

    def __init__(self, args):
        """For datasets, they should have been processed into groups beforehand.
        """
        self.args = args
        self.PERFORM_CV  = args.do_cv
        self.NET_TYPE    = args.net_type
        self.PRINT_PREDS = args.print_preds
        self.CONFIG_NAME = 'grasp'
        self.ROOT_DIR    = '/nfs/diskstation/seita/bed-make/'   # Tritons
        self.DATA_PATH   = self.ROOT_DIR+''                     # Tritons

        # If `PERFORM_CV` then we are training on a set of cached, separate data.
        # We split data in the `data_manager` class, so OK to load all groups here.
        # Else, train on all the cached data, +OPTIONALLY a new held-out test set.
        if self.PERFORM_CV:
            self.CV_HELD_OUT_INDEX = args.cv_idx
            assert args.cv_idx is not None

        self.ROLLOUT_PATH = join(self.DATA_PATH, 'cache_combo_v01/')
        self.CV_GROUPS = sorted(
                [x for x in os.listdir(self.ROLLOUT_PATH) if 'cv_' in x]
        )
        assert len(self.CV_GROUPS) == 10

        # To ignore a test set, set as None. Else, load in all the groups.
        # (We also have test set saved as CV splits, but we'll load everything.)
        #self.TEST_ROLLOUT_PATH = join(self.DATA_PATH, 'cache_white_v01/')
        self.TEST_ROLLOUT_PATH = None
        if self.TEST_ROLLOUT_PATH is not None:
            self.TEST_GROUPS = sorted(
                    [x for x in os.listdir(self.TEST_ROLLOUT_PATH) if 'cv_' in x]
            )
        self.HAVE_TEST_SET = (self.PERFORM_CV or self.TEST_ROLLOUT_PATH is not None)

        # Training info (stats, checkpoints, etc.), goes here for grasp net.
        self.OUT_DIR = join(self.DATA_PATH, 'grasp/')

        # Pre-trained weights.
        self.WEIGHTS_DIR = join(self.DATA_PATH,'weights/')
        self.PRE_TRAINED_DIR = \
            '/nfs/diskstation/seita/yolo_tensorflow/data/pascal_voc/weights/'
        self.WEIGHTS_FILE = None

        # Classes, labels, data augmentation
        self.CLASSES = ['success_grasp','fail_grasp']

        # Model parameters. The USE_DEPTH is a critical one to test!
        self.T_IMAGE_SIZE_H = 480
        self.T_IMAGE_SIZE_W = 640
        if args.shrink_images:
            assert args.use_smaller_net and (not args.fix_pretrained_layers) \
                    and args.net_type == 4
            self.IMAGE_SIZE = 224
        else:
            self.IMAGE_SIZE = 448
        self.CELL_SIZE = 7
        self.BOXES_PER_CELL = 2
        self.ALPHA = 0.1
        self.DISP_CONSOLE = True
        self.RESOLUTION = 10
        self.DROPOUT_KEEP_PROB = args.dropout_keep_prob
        self.L2_LAMBDA = args.l2_lambda

        # IMPORTANT !!!!!!! False means RGB, which we normally DON'T want.
        self.USE_DEPTH = True

        # solver parameter
        # ----------------------------------------------------------------------
        # If 'fixing', this means 'images' effectively turn into (fs,fs,channels),
        # e.g., shape (14,14,1024). If False, train from normal (480,640,3)-size data.
        # Technically 'pre-trained' is more like 'pre-initialized' (from Pascal task).
        # Update: if we use a smaller network, `FIX_PRETRAINED_LAYERS==False`.
        # ----------------------------------------------------------------------
        self.FIX_PRETRAINED_LAYERS = args.fix_pretrained_layers
        self.SMALLER_NET = args.use_smaller_net
        assert not (self.SMALLER_NET and self.FIX_PRETRAINED_LAYERS)

        self.OPT_ALGO = 'ADAM'
        if self.OPT_ALGO == 'ADAM':
            if args.lrate >= 0.01:
                print("Don't run Adam with high LR")
                sys.exit()
            self.LEARNING_RATE = args.lrate
            self.USE_EXP_MOV_AVG = False
        elif self.OPT_ALGO == 'SGD':
            self.LEARNING_RATE = args.lrate
            self.USE_EXP_MOV_AVG = True
        else:
            raise ValueError(self.OPT_ALGO)

        # Decay LR every `DECAY_STEPS` but probably easiest to keep a fixed LR.
        self.DECAY_STEPS = 10000
        self.DECAY_RATE = 0.1
        self.STAIRCASE = True
        self.BATCH_SIZE = args.batch_size
        self.MAX_ITER = args.max_iters
        self.SUMMARY_ITER = 1
        self.TEST_ITER = 1
        self.SAVE_ITER = 500
        self.VIZ_DEBUG_ITER = 400
        self.GPU_MEM_FRAC = args.gpu_frac

        # fast params
        self.FILTER_SIZE = 14
        self.NUM_FILTERS = 1024


    def compute_label(self,datum):
        """Labels scaled in [-1,1], as described in paper.

        Actually, [-0.5, 0.5]. I thought we needed to further adjust for the re-sizing
        of the image to (448,448), but turns out this is still valid ... since the re-
        sized value cancels out. Huh, interesting.
        """
        pose = datum['pose']
        label = np.zeros((2))
        x = (float(pose[0]) / self.T_IMAGE_SIZE_W) - 0.5
        y = (float(pose[1]) / self.T_IMAGE_SIZE_H) - 0.5
        label = np.array([x,y])
        return label


    def compare_preds_labels(self, preds, labels, doprint=False):
        """Try to get losses in the original pixel space for interpretability."""
        xx = np.array([self.T_IMAGE_SIZE_W, self.T_IMAGE_SIZE_H])
        raw_preds  = (0.5 + preds) * xx
        raw_labels = (0.5 + labels) * xx
        assert np.min(raw_labels) > 0.0
        delta = raw_preds - raw_labels # shape (batchsize,2)
        test_loss_raw = np.mean( np.linalg.norm(delta, axis=1) )
        if doprint:
            print("grasp test preds:\n{}".format(preds))
            print("grasp test labels:\n{}".format(labels))
            print("(raw) grasp test preds:\n{}".format(raw_preds))
            print("(raw) grasp test labels:\n{}".format(raw_labels))
        return test_loss_raw


    def return_raw_labels(self, arr):
        """Assumes `arr` is like the logits or predictions for scaled labels."""
        xx = np.array([self.T_IMAGE_SIZE_W, self.T_IMAGE_SIZE_H])
        raw = (0.5 + arr) * xx
        minv = np.min(raw)
        if minv <= 0.0:
            print("Caution: our net predicted min pixel value: {}".format(minv))
        return raw


    def get_empty_state(self, batchdim=None):
        """Each time we call a batch during training/testing, init w/this."""
        bs = self.BATCH_SIZE
        if batchdim is not None:
            bs = batchdim

        if self.FIX_PRETRAINED_LAYERS:
            assert not self.SMALLER_NET and self.NET_TYPE != 2
            return np.zeros((bs, self.FILTER_SIZE, self.FILTER_SIZE, self.NUM_FILTERS))
        else:
            assert self.NET_TYPE != 1
            return np.zeros((bs, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))


    def get_empty_label(self, batchdim=None):
        """Each time we call a batch during training/testing, init w/this."""
        if batchdim is not None:
            return np.zeros((batchdim, 2))
        else:
            return np.zeros((self.BATCH_SIZE, 2))
