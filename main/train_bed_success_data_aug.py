import tensorflow as tf
import datetime
import os
import argparse

from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.data_manager import data_manager
from fast_grasp_detect.core.train_network import Solver


import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from fast_grasp_detect.configs.bed_success_config import CONFIG

# Data augmentation transformations to apply.
ORIGINAL = 'original'
GAUSSIAN = 'gaussian'
SALT_PEPPER = 'salt_pepper'
HIST_EQUALIZATION = 'hist_equalization'
LIGHTING = 'lighting'
HOR_FLIPS = 'hor_flips'
VERT_FLIPS = 'vert_flips'
FANCY_PCA = 'fancy_pca'

def modify_config(cfg, cfg_name):
    """
    Modifies the existing config to apply the right data augmentation technique (if any).
    Params:
        cfg: Config object to modify.
        cfg_name: Type of data augmentation technique to apply.
    Returns:
        The modified config.
    """
    curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    # Append timestamp to config names to prevent overwriting existing files.
    cfg.CONFIG_NAME = cfg_name + curr_time
    
    # Disable all data augmentation by default.
    cfg.VERT_FLIP = False
    cfg.HOR_FLIP = False
    cfg.LIGHTING_NOISE = False
    cfg.GAUSSIAN_NOISE = False
    cfg.SALT_PEPPER_NOISE = False
    cfg.HIST_EQUALIZATION = False
    cfg.FANCY_PCA = False

    if cfg_name == GAUSSIAN:
        cfg.GAUSSIAN_NOISE = True
    elif cfg_name == SALT_PEPPER:
        cfg.SALT_PEPPER_NOISE = True
    elif cfg_name == HIST_EQUALIZATION:
        cfg.HIST_EQUALIZATION = True
    elif cfg_name == LIGHTING:
        cfg.LIGHTING_NOISE = True
    elif cfg_name == VERT_FLIPS:
        cfg.VERT_FLIP = True
    elif cfg_name == HOR_FLIPS:
        cfg.HOR_FLIP = True
    elif cfg_name == FANCY_PCA:
        cfg.FANCY_PCA = True

    return cfg

noise_types = [ORIGINAL, GAUSSIAN, SALT_PEPPER, HIST_EQUALIZATION, LIGHTING, HOR_FLIPS, VERT_FLIPS]

for noise_type in noise_types:
    bed_success_options = modify_config(CONFIG(), noise_type)

    pascal = data_manager(bed_success_options)

    yolo = SNet(bed_success_options)


    solver = Solver(bed_success_options,yolo,pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')
