import tensorflow as tf
import datetime
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver

from fast_grasp_detect.configs.tl_make_rollouts_dir import *

import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

rollout_parent_path = '/media/autolab/1tb/data/bed_rcnn/rollouts_tl/'

from fast_grasp_detect.configs.bed_grasp_config import CONFIG

def main():
    num_cal_data = [i*5 for i in range(9)]
    ts = make_training_sets(num_cal_data)
    test_losses = []

    for trial in num_cal_data:
        rollout_dir = rollout_parent_path + 'rollouts_tl_' + str(trial) + '/'

        assert os.path.exists(rollout_dir), "\nNo such rollout directory.\n"
        bed_grasp_options = CONFIG(rollout_path=rollout_dir)
        pascal = data_manager(bed_grasp_options)
        yolo = GHNet(bed_grasp_options)

        solver = Solver(bed_grasp_options,yolo,pascal)

        print('Start training ...')
        best_test_loss = solver.train()
        test_losses.append(best_test_loss)
        print('Done training.')

    print("\n\n\nTest losses:")
    print(test_losses)

    plt.plot(num_cal_data, test_losses)
    plt.xlabel('Number of Cal Training Points')
    plt.ylabel('Minimum Test Loss')
    plt.show()


if __name__ == "__main__": main()
