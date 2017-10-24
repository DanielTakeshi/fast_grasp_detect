import tensorflow as tf
import datetime
import os
import argparse

from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver


import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from fast_grasp_detect.configs.bed_grasp_config import CONFIG

bed_grasp_options = CONFIG()

pascal = data_manager(bed_grasp_options)

yolo = GHNet(bed_grasp_options)


solver = Solver(bed_grasp_options,yolo,pascal)

print('Start training ...')
solver.train()
print('Done training.')
