import tensorflow as tf
import datetime, os, time, random, argparse
import numpy as np
from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver
from fast_grasp_detect.configs.bed_success_config import CONFIG


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
pp.add_argument('--batch_size', type=int, default=32,
        help='Usual batch size, w/64 it may result in lots of memory allocation')

# Booleans. For network design we'll use an integer and then pick values.
pp.add_argument('--do_cv', action='store_true', default=False,
        help='If not cv, then we may or may not have (held-out) test data')
pp.add_argument('--print_preds', action='store_true', default=False,
        help='For printing actual predictions during validation. I\'d avoid for now.')
pp.add_argument('--net_type', type=int,
        help='1=yolo-fix26, 2=yolo-all, 3=a-net 448x448, 4=a-net 227x227')

args = pp.parse_args()
set_seed(args.seed)

# Add stuff here, makes selecting net design less error-prone.
if int(args.net_type) == 1:
    args.fix_pretrained_layers = True
    args.use_smaller_net = False
    args.shrink_images = False
elif int(args.net_type) == 2:
    args.fix_pretrained_layers = False
    args.use_smaller_net = False
    args.shrink_images = False
elif int(args.net_type) == 3:
    args.fix_pretrained_layers = False
    args.use_smaller_net = True
    args.shrink_images = False
elif int(args.net_type) == 4:
    args.fix_pretrained_layers = False
    args.use_smaller_net = True
    args.shrink_images = True
else:
    raise ValueError(args.net_type)

# Config, then three major components; `pascal` has reference to 'yolo' network.
bed_success_options = CONFIG(args)
pascal = data_manager(bed_success_options)
yolo = SNet(bed_success_options, pascal.yc)
solver = Solver(bed_success_options, yolo, pascal)

inspect_variables()
print('\nStart training ...')
solver.train()
print('Done training.')
