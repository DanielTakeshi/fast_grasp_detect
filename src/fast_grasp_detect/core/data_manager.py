import os
import xml.etree.ElementTree as ET
import numpy as np
from numpy.random import random
import cv2
import cPickle
import copy
import glob


from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.data_aug.data_augment import augment_data
from fast_grasp_detect.data_aug.fancy_pca import compute_covariance_matrix


import cPickle as pickle
import IPython

class data_manager(object):
    def __init__(self,options):

        self.cfg = options

        self.rollout_path = self.cfg.ROLLOUT_PATH
        self.batch_size = self.cfg.BATCH_SIZE
       
        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE


        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        
        # Removing these for now. The entire config gets passed to the data
        # augmentation library, which is used to determine what transformations
        # to apply.
        # self.flipped = self.cfg.FLIPPED
        # self.noise = self.cfg.LIGHTING_NOISE 

        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.test_labels = None

        # Percentange of training examples to hold out for cross-validation.
        self.holdout_percentage = self.cfg.HOLDOUT_PERCENTAGE


        self.yc = YOLO_CONV(self.cfg)
        self.yc.load_network()


        self.recent_batch = []
        self.load_test_set()
        self.load_rollouts()

    def get(self, noise=False):
        images = self.cfg.get_batch_empty_state()
        labels = self.cfg.get_batch_empty_label()
        count = 0
        self.recent_batch = []
        while count < self.batch_size:
            
            images[count, :, :, :] = self.train_labels[self.cursor]['features']
           
            labels[count, :] = self.train_labels[self.cursor]['label']

            self.recent_batch.append(self.train_labels[self.cursor])

            count += 1
            if(count == self.batch_size):
                break

            self.cursor += 1
            if self.cursor >= len(self.train_labels):
                np.random.shuffle(self.train_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def get_test(self):
        num_test = len(self.test_labels)

        images = self.cfg.get_num_empty_state(num_test)
        labels = self.cfg.get_num_empty_label(num_test)

        count = 0

        for i in range(num_test):
            images[count, :, :, :] = self.test_labels[i]['features']
            labels[count, :] = self.test_labels[i]['label']

        return images, labels

    def get_val(self):
        num_val = len(self.val_labels)

        images = self.cfg.get_num_empty_state(num_val)
        labels = self.cfg.get_num_empty_label(num_val)

        count = 0

        for i in range(num_val):
            images[count, :, :, :] = self.val_labels[i]['features']
            labels[count, :] = self.val_labels[i]['label']
        
        return images, labels
    
    def prep_image(self, image):
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        return image

    def break_up_rollouts(self,rollout):

        grasp_point = []
        grasp_rollout = []
        for data in rollout:
            
            
            if type(data) == list:
                continue

            if(data['type'] == 'grasp'):
                grasp_point.append(data)

            elif(data['type'] == 'success'):
                if( len(grasp_point) > 0):
                    grasp_rollout.append(grasp_point)
                    grasp_point = []

        return grasp_rollout


    def load_test_set(self):

        self.test_labels = []

        self.train_data_path = []
        self.test_data_path = []
        rollouts = glob.glob(os.path.join(self.cfg.BC_HELD_OUT, '*_*'))

        count = 0
        
        for rollout_p in rollouts:
            #rollout_p = rollouts[0]  
            rollout = pickle.load(open(rollout_p+'/rollout.p'))


            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            for grasp_point in grasp_rollout:
                    print "TEST EXAMPLE", rollout_p
                    
                    features = self.yc.extract_conv_features(grasp_point[0]['c_img'])

                    label = self.cfg.compute_label(grasp_point[0])
                    self.test_labels.append({'c_img': grasp_point[0]['c_img'], 'label': label, 'features':features})
                    self.test_data_path.append(rollout_p)


 


    def load_rollouts(self):
       
        self.train_labels = []
      

        self.train_data_path = []
        self.test_data_path = []
        rollouts = glob.glob(os.path.join(self.rollout_path, '*_*'))

        count = 0

        # If Fancy PCA data augmentation technique is enabled, we need to
        # use PCA on the entire dataset to get the RGB directions to use
        # in the actual data augmentation later on.
        cov = None
        if self.cfg.FANCY_PCA:
            imgs = []
            for rollout_p in rollouts:
                rollout = pickle.load(open(rollout_p+'/rollout.p'))

                grasp_rollout = self.cfg.break_up_rollouts(rollout)
                for grasp_point in grasp_rollout:
                    for data in grasp_point:
                        imgs.append(data['c_img'])
            cov = compute_covariance_matrix(imgs)
        

        for rollout_p in rollouts:
            #rollout_p = rollouts[0]  
            rollout = pickle.load(open(rollout_p+'/rollout.p'))

          
            print rollout_p
            print len(rollout)

            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            for grasp_point in grasp_rollout:
            
            
                count = 0
                
                for data in grasp_point:
                    
                    data_a = augment_data(data, self.cfg, cov)
                    
                    for datum_a in data_a:
                        
                        features = self.yc.extract_conv_features(datum_a['c_img'])

                        label = self.cfg.compute_label(datum_a)

                        self.train_labels.append({'c_img': datum_a['c_img'], 'label': label, 'features':features})
                self.train_data_path.append(rollout_p)

        # Reserve some holdout data for cross-validation.
        np.random.shuffle(self.train_labels)
        num_holdout = int(len(self.train_labels) * self.holdout_percentage)
        self.train_labels = self.train_labels[:num_holdout]
        self.val_labels = self.train_labels[num_holdout:]
      
        return 

 


    def compute_label(self, pose):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        label = np.zeros((2))

        x = pose[0]/cfg.T_IMAGE_SIZE_W-0.5
        y = pose[1]/cfg.T_IMAGE_SIZE_H-0.5

        label = np.array([x,y])


        

        return label


     
        

