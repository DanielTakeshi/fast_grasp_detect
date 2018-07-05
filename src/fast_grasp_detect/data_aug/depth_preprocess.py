# From this commit:
# https://github.com/mdlaskey/fast_grasp_detect/commit/2f85441c86fa7eed089cafb638f0a5bb2fa1eddb
#
# and then this to fix a few typos, etc.:
# https://github.com/mdlaskey/fast_grasp_detect/commit/1004c2f084a16b16344b6d6016efee283bac17ae

import numpy as np


def depth_to_3ch(img):
    new_img = np.zeros([img.shape[0],img.shape[1],3])
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def depth_scaled_to_255(img):
    img = 255.0/np.max(img)*img
    return img


def depth_to_net_dim(img):
    img = depth_to_3ch(img)
    img = depth_scaled_to_255(img)
    return img


def datum_to_net_dim(datum):
    """ (480,640) -> (480,640,3) """
    datum['d_img'] = depth_to_net_dim(datum['d_img'])
