import tensorflow as tf
import datetime
import os
import argparse
import sys

from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver


import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

rollout_parent_path = '/media/autolab/1tb/data/bed_rcnn/rollouts_tl/'

from fast_grasp_detect.configs.bed_grasp_config import CONFIG

def main():
    rollout_num = sys.argv[1]
    rollout_dir = rollout_parent_path + 'rollouts_tl_' + str(rollout_num) + '/'
    assert os.path.exists(rollout_dir), "\nNo such rollout directory.\n"
    bed_grasp_options = CONFIG(rollout_path=rollout_dir)
    pascal = data_manager(bed_grasp_options)
    yolo = GHNet(bed_grasp_options)

    solver = Solver(bed_grasp_options,yolo,pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')
