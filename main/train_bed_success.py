import tensorflow as tf
import datetime, os, time, random
import numpy as np
from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver
import IPython
import cPickle as pickle
slim = tf.contrib.slim
from fast_grasp_detect.configs.bed_success_config import CONFIG

bed_success_options = CONFIG()
tf.set_random_seed(bed_success_options.SEED)
np.random.seed(bed_success_options.SEED)
random.seed(bed_success_options.SEED)

pascal = data_manager(bed_success_options)
yolo = SNet(bed_success_options)
solver = Solver(bed_success_options,yolo,pascal)

print('\nJust before training, here is `tf.GraphKeys.TRAINABLE_VARIABLES`:')
variables = tf.trainable_variables()
for vv in variables:
    print(vv)
print('\nAnd here is `tf.GraphKeys.GLOBAL_VARIABLES`:')
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for vv in variables:
    print(vv)

print('\nStart training ...')
solver.train()
print('Done training.')
