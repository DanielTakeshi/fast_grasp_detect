import tensorflow as tf
import datetime
import os, time
import argparse
from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver
import IPython
import cPickle as pickle
slim = tf.contrib.slim
from fast_grasp_detect.configs.bed_success_config import CONFIG

bed_success_options = CONFIG()
pascal = data_manager(bed_success_options)
yolo = SNet(bed_success_options)
solver = Solver(bed_success_options,yolo,pascal)

print('\nJust before training, here are all the TF variables we have:')
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for vv in variables:
    print(vv)

print('\nStart training ...')
solver.train()
print('Done training.')
