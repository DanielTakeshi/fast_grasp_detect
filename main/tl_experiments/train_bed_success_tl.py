import tensorflow as tf
import datetime
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver


import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

rollout_parent_path = '/media/autolab/1tb/data/bed_rcnn/rollouts_tl/'

from fast_grasp_detect.configs.bed_success_config import CONFIG

def main():
    num_cal_data = [i*5 for i in range(9)]
    # ts = make_training_sets(num_cal_data)
    test_losses = []

    for trial in num_cal_data:
        rollout_dir = rollout_parent_path + 'rollouts_tl_' + str(trial) + '/'

        assert os.path.exists(rollout_dir), "\nNo such rollout directory.\n"
        bed_success_options = CONFIG(rollout_path=rollout_dir)
        bed_success_options.CONFIG_NAME = 'success_net_tl_' + str(trial)
        pascal = data_manager(bed_success_options)
        yolo = SNet(bed_success_options)
    
        solver = Solver(bed_success_options,yolo,pascal)

        print('Start training ...')
        best_test_loss, ckpt = solver.train()
        test_losses.append(best_test_loss)
        print("Saved to: " + str(ckpt))
        print('Done training.')
        tf.reset_default_graph()

    print("\n\n\nTest losses:")
    print(test_losses)

    fig = plt.figure()
    plt.plot(num_cal_data, test_losses)
    plt.xlabel('Number of Cal Training Points')
    plt.ylabel('Minimum Test Loss - Success Net')
    # plt.show()
    fig.savefig('success_tl.png')



if __name__ == "__main__": main()