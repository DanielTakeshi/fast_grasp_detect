import tensorflow as tf
import datetime
import os, time
import argparse
from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver
import IPython
import cPickle as pickle
slim = tf.contrib.slim
from fast_grasp_detect.configs.bed_grasp_config import CONFIG

bed_grasp_options = CONFIG()
pascal = data_manager(bed_grasp_options)
yolo = GHNet(bed_grasp_options)
solver = Solver(bed_grasp_options,yolo,pascal)

print('\nJust before training, here is `tf.GraphKeys.TRAINABLE_VARIABLES`:')
variables = tf.trainable_variables()#get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for vv in variables:
    print(vv)
print('\nAnd here is `tf.GraphKeys.GLOBAL_VARIABLES`:')
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for vv in variables:
    print(vv)

print('\nStart training ...')
solver.train()
print('Done training.')
