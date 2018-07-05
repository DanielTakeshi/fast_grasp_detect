# From this commit:
# https://github.com/mdlaskey/fast_grasp_detect/commit/2f85441c86fa7eed089cafb638f0a5bb2fa1eddb
#
# and then this to fix a few typos, etc.:
# https://github.com/mdlaskey/fast_grasp_detect/commit/1004c2f084a16b16344b6d6016efee283bac17ae
#
# then finally:
# https://github.com/mdlaskey/fast_grasp_detect/commit/a87af292c3d92169551cda534f3844151f782a00

import numpy as np
import IPython
import cv2


def depth_to_3ch(img):
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    img = img.flatten()
    img[img>1000] = 0
    img = img.reshape([w,h])
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def depth_scaled_to_255(img):
    img = 255.0/np.max(img)*img
    img = np.array(img,dtype=np.uint8)
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img


def depth_to_net_dim(img):
    img = depth_to_3ch(img)
    img = depth_scaled_to_255(img)
    return img


def datum_to_net_dim(datum):
    """ (480,640) -> (480,640,3) """
    datum['d_img'] = depth_to_net_dim(datum['d_img'])
    return datum
