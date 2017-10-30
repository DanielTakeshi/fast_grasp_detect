import cv2
import cPickle as pickle
import IPython
import numpy as np

from fast_grasp_detect.data_aug.augment_lighting import addGaussianNoise
from fast_grasp_detect.data_aug.augment_lighting import addSaltPepperNoise
from fast_grasp_detect.data_aug.augment_lighting import equalizeHistRGB
from fast_grasp_detect.data_aug.augment_lighting import get_lighting

import copy
HALF_LENGTH = 15

RADIUS = 10

THICKNESS = 1
C_THICKNESS = 1

COLOR = (0,0,255) #RGB


def flip_data_vertical(img, label, clss):

    h,w,channel = img.shape

    v_img = cv2.flip(img,1)

    label[0] = w-label[0]

    return {'c_img': v_img, 'pose': label, 'class': clss}


def flip_data_horizontal(img, label, clss):

    h,w,channel = img.shape

    h_img = cv2.flip(img,0)

    label[1] = h-label[1]

    return {'c_img': h_img, 'pose': label, 'class': clss}


def augment_data(data, cfg):
    """
    Creates a list of augmented training examples.
    Params:
        data: original example consisting of an RGB color image, a pose,
            and a class label.
        cfg: a configuration object that should contain options for which
            data augmentation techniques to apply.
    Returns:
        A list of training examples, consisting of applying various data
        augmentation techniques as specified in the config to the original
        example.
    """
    augmented_data = []

    img = data['c_img']
    label = data['pose']
    clss = data['class']

    if label == None: 
        label = [0,0]
    
    # Create list of images from applying various lighting data augmentation
    # techniques. Includes the original image.
    augmented_lighting_imgs = [img]

    # Applies various lighting filters.
    if cfg.LIGHTING_NOISE:
        augmented_lighting_imgs += get_lighting(img)

    # Adds i.i.d. Gaussian noise to each pixel in each channel.
    if cfg.GAUSSIAN_NOISE:
        augmented_lighting_imgs.append(addGaussianNoise(img))

    # Chooses pixels at random to set to white or black.
    if cfg.SALT_PEPPER_NOISE:
        augmented_lighting_imgs.append(addSaltPepperNoise(img))

    # Modifies image to take on full range of values in each channel.
    if cfg.HIST_EQUALIZATION:
        augmented_lighting_imgs.append(equalizeHistRGB(img))

    for lighting_img in augmented_lighting_imgs:
        # No orientation changes.
        p_n = {'c_img': lighting_img, 'pose': label, 'class': clss}
        augmented_data.append(p_n)

        # Perform vertical flip of the image, modifying label accordingly.
        if cfg.VERT_FLIP:
            p_v = flip_data_vertical(lighting_img, np.copy(label), clss)
            augmented_data.append(p_v)

        # Perform horizontal flip of the image, modifying label accordingly.
        if cfg.HOR_FLIP:
            p_h = flip_data_horizontal(lighting_img, np.copy(label), clss)
            augmented_data.append(p_h)


    return augmented_data



if __name__ == "__main__":


    dp = DrawPrediction()

    path = cfg.ROLLOUT_PATH+'rollout_0/rollout.p'
    data = pickle.load(open(path,'rb'))

    grasp_point = data[0]

    box = grasp_point['label']['objects'][0]['box']

    x = int((box[0] + box[2])/2.0) 

    y = int((box[1]+ box[3])/2.0)

    pose = [x,y]



    c_img = grasp_point['c_img']
    


    image = dp.draw_prediction(c_img,pose)

    cv2.imshow('debug', image)
    cv2.waitKey(0)

    print "RESULT ", sc.check_success(wl)