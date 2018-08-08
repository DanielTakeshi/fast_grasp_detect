import tensorflow as tf
import datetime, os, time, random, argparse
import numpy as np
from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver
from fast_grasp_detect.configs.bed_grasp_config import CONFIG


def set_seed(s):
    tf.set_random_seed(s)
    np.random.seed(s)
    random.seed(s)


def inspect_variables():
    print('\nJust before training, here is `tf.trainable_variables()`:')
    numv = 0
    variables = tf.trainable_variables()
    for vv in variables:
        numv += np.prod(vv.shape)
        print(vv)
    print("\nNumber of trainable variables/parameters: {}".format(numv))


pp = argparse.ArgumentParser()

# Various hyperparameters
pp.add_argument('--seed', type=int, default=0,
        help='Usual random seed for training.')
pp.add_argument('--max_iters', type=int, default=100,
        help='Maximum iterations. Unfortunately not as simple as an epoch')
pp.add_argument('--cv_idx', type=int,
        help='Easiest to iterate through this in a bash script')
pp.add_argument('--lrate', type=float, default=0.00010,
        help='Default _starting_ LR. Assumes Adam, if SGD then increase LR.')
pp.add_argument('--dropout_keep_prob', type=float, default=1.0,
        help='The dropout keep probability. 1.0 means no dropout.')
pp.add_argument('--l2_lambda', type=float, default=0.00010,
        help='Standard L2 regularization term.')
pp.add_argument('--gpu_frac', type=float, default=0.75,
        help='Use a value less than 0.9 to leave some memory available.')

# Booleans. A bit awkward for neural net design choice because the default choice
# (without specifying anything) is with the YOLO net but training all the layers...
pp.add_argument('--do_cv', action='store_true', default=False,
        help='If not doing cv, then assumes we have a fixed held-out directory of test data')
pp.add_argument('--use_smaller_net', action='store_true', default=False,
        help='Use this to avoid the YOLO stem')
pp.add_argument('--fix_pretrained_layers', action='store_true', default=False,
        help='If using YOLO stem, this is usually a good idea')
pp.add_argument('--print_preds', action='store_true', default=False,
        help='For printing actual predictions during validation. I\'d avoid this for now.')
pp.add_argument('--use_cache', action='store_true', default=False,
        help='Use this if possible to avoid loading rollouts from /nfs/diskstation.')

args = pp.parse_args()
set_seed(args.seed)

# Configuration, then three major components; `pascal` has reference to 'yolo' network.
bed_grasp_options = CONFIG(args)
pascal = data_manager(bed_grasp_options)
yolo = GHNet(bed_grasp_options, pascal)
solver = Solver(bed_grasp_options, yolo, pascal)

inspect_variables()
print('\nStart training ...')
solver.train()
print('Done training.')
