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


def depth_to_3ch(img, cutoff):
    """Careful if the cutoff is in meters or millimeters!
    It's useful to turn the background into black into the depth images.
    """
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    img = img.flatten()
    img[img>cutoff] = 0.0
    img = img.reshape([w,h])
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def depth_scaled_to_255(img):
    assert np.max(img) > 0.0
    img = 255.0/np.max(img)*img
    img = np.array(img,dtype=np.uint8)
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img


def depth_to_net_dim(img, robot):
    """This should be the ONLY place we set the cutoff!!!"""
    assert robot in ['Fetch', 'HSR']
    if robot == 'HSR':
        cutoff = 1400
    elif robot == 'Fetch':
        cutoff = 1.4
    img = depth_to_3ch(img, cutoff)
    img = depth_scaled_to_255(img)
    return img


def datum_to_net_dim(datum, robot):
    """ (480,640) -> (480,640,3) """
    assert robot in ['Fetch', 'HSR']
    datum['d_img'] = depth_to_net_dim(datum['d_img'], robot)
    return datum
