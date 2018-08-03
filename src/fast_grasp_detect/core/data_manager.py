import os, cv2, sys, copy, glob, IPython
import xml.etree.ElementTree as ET
import numpy as np
from numpy.random import random
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.data_aug.data_augment import augment_data
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
import cPickle as pickle


class data_manager(object):

    def __init__(self,options):
        self.cfg = options
        self.rollout_path = self.cfg.ROLLOUT_PATH
        self.batch_size = self.cfg.BATCH_SIZE
        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        #self.flipped = self.cfg.FLIPPED
        #self.noise = self.cfg.LIGHTING_NOISE
        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1
        self.gt_labels = None

        # For cross-validation
        if self.cfg.PERFORM_CV:
            self.held_out_list = self.cfg.CV_GROUPS[self.cfg.CV_HELD_OUT_INDEX]
            self.held_out_rollouts = [os.path.join(self.rollout_path,'rollout_'+str(rr)) for rr in self.held_out_list]

        # Load YOLO network and set up its pre-trained weights from known file
        print("\n`data_manager` class, now calling YOLO_CONV and loading network...")
        self.yc = YOLO_CONV(self.cfg)
        self.yc.load_network()

        # Load test set and training rollouts. Also make test set batch fixed (shouldn't be re-shuffling).
        self.recent_batch = []
        self.test_batch_images = None
        self.test_batch_labels = None
        self.load_test_set()
        self.load_rollouts()


    def get(self, noise=False):
        """Creates and returns images/labels for training, where the images are features from the
        pre-trained YOLO network, or at least processed to correct sizes.

        `count` ensures we match batch size, `cursor` since it may overlap w/end of the epoch.
        """
        images = self.cfg.get_empty_state()
        labels = self.cfg.get_empty_label()
        count = 0
        self.recent_batch = []

        while count < self.batch_size:
            images[count, :, :, :] = self.train_labels[self.cursor]['features']
            labels[count, :] = self.train_labels[self.cursor]['label']
            self.recent_batch.append(self.train_labels[self.cursor])
            count += 1
            if (count == self.batch_size):
                break
            self.cursor += 1
            # After we've gone through the data, re-shuffle it.
            if self.cursor >= len(self.train_labels):
                np.random.shuffle(self.train_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels


    def get_test(self):
        """Creates and returns images/labels for testing, where the images are
        features from the pre-trained YOLO network.

        Unlike the training case, here we should just take the entire test data in one batch.
        """
        assert self.test_batch_images is not None and self.test_batch_labels is not None
        return (self.test_batch_images, self.test_batch_labels)


    def prep_image(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0
        return image


    def load_test_set(self):
        """ Assigns to `self.test_labels`, each element a dict w/relevant info.

        If we are doing cross validation to evaluate different training criteria before hand, we'll
        use cross validation groups and set the appropriate held-out dataset with our config file.

        Also, form the test batch at the end. It won't be large with the data we have and we
        shouldn't keep re-computing and re-shuffling since it's a test (or validation) batch.
        """
        if self.cfg.PERFORM_CV:
            print("\ndata_manager.load_test_set(), held-out rollouts: {} (cv index {}) from {}".format(
                    self.held_out_list, self.cfg.CV_HELD_OUT_INDEX, self.rollout_path))
            rollouts = list(self.held_out_rollouts)
        else:
            print("\ndata_manager.load_test_set(), path: {}".format(self.cfg.BC_HELD_OUT))
            rollouts = glob.glob(os.path.join(self.cfg.BC_HELD_OUT, '*_*'))

        self.test_labels = []

        # Oops, bad naming, `grasp_rollout` could also contain the success, but w/e.
        for rollout_p in rollouts:
            rollout = pickle.load(open(rollout_p+'/rollout.p'))
            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            print("{},  len(grasp_rollout)={},  w/len(rollout)={} [TEST]".format(
                    rollout_p, len(grasp_rollout), len(rollout)))

            for grasp_point in grasp_rollout:
                # Run the YOLO network w/pre-trained weights!!
                if self.cfg.USE_DEPTH:
                    grasp_point[0] = datum_to_net_dim(grasp_point[0])
                    features = self.yc.extract_conv_features(grasp_point[0]['d_img'])
                else:
                    features = self.yc.extract_conv_features(grasp_point[0]['c_img'])
                label = self.cfg.compute_label(grasp_point[0])
                data_pt = {'c_img':grasp_point[0]['c_img'], 'label':label, 'features':features}
                self.test_labels.append(data_pt)

        # Form and investigate the testing images and labels in their batch.
        K = len(self.test_labels)
        print("len(self.test_labels): {}".format(K))
        assert K <= 500
        self.test_batch_images = self.cfg.get_empty_state(batchdim=K)
        self.test_batch_labels = self.cfg.get_empty_label(batchdim=K)
        for count in range(K):
            self.test_batch_images[count, :, :, :] = self.test_labels[count]['features']
            self.test_batch_labels[count, :]       = self.test_labels[count]['label']
        print("test_batch_images.shape: {}".format(self.test_batch_images.shape))
        print("test_batch_labels.shape: {}".format(self.test_batch_labels.shape))
        print("test_batch_labels:\n{}".format(self.test_batch_labels))
        if self.cfg.CONFIG_NAME == 'grasp_net':
            W, H = self.cfg.T_IMAGE_SIZE_W, self.cfg.T_IMAGE_SIZE_H
            raw_labels = (self.test_batch_labels + 0.5) * np.array([W,H])
            print("(raw) test_batch_labels:\n{}".format(raw_labels))


    def load_rollouts(self):
        """ Assigns to `self.train_labels`, each element a dict w/relevant info.

        Each rollout in the designated directory is broken up into 'grasp_point' dicts, which are
        like the data dicts we used for collecting data (though w/out 'd_img'). Then for each, we
        augment the data and then run the YOLO network on that to get pre-extracted features.
        """
        if self.cfg.PERFORM_CV:
            print("\ndata_manager.load_rollouts(), path {} (but w/held-out ignored: {}, index {})".format(
                    self.rollout_path, self.held_out_list, self.cfg.CV_HELD_OUT_INDEX))
            rollouts = [rr for rr in glob.glob(os.path.join(self.rollout_path, '*_*'))
                        if rr not in self.held_out_rollouts]
        else:
            print("\ndata_manager.load_rollouts(), path: {}".format(self.rollout_path))
            rollouts = glob.glob(os.path.join(self.rollout_path, '*_*'))

        self.train_labels = []

        for rollout_p in rollouts:
            rollout = pickle.load(open(rollout_p+'/rollout.p'))
            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            print("{},  len(grasp_rollout)={},  w/len(rollout)={} [TEST]".format(
                    rollout_p, len(grasp_rollout), len(rollout)))

            for grasp_point in grasp_rollout:
                for data in grasp_point:
                    if self.cfg.USE_DEPTH:
                        data = datum_to_net_dim(data)
                    data_a = augment_data(data, self.cfg.USE_DEPTH) # data augmentation magic.
                    for d_idx,datum_a in enumerate(data_a):
                        # run the YOLO network w/pre-trained weights! 
                        # features.shape: (1, 14, 14, 1024)
                        # labels: for grasps, alternate between (x,y) and (-x,y)
                        features = self.yc.extract_conv_features(datum_a['c_img'])
                        label = self.cfg.compute_label(datum_a)
                        data_pt = {'c_img':datum_a['c_img'],'label': label,'features': features}
                        self.train_labels.append(data_pt)

        np.random.shuffle(self.train_labels)
        print("len(self.train_labels): {}. Also, shuffled!".format(len(self.train_labels)))


    def compute_label(self, pose):
        """Load image and bounding boxes info from XML file in the PASCAL VOC format."""
        label = np.zeros((2))
        x = pose[0]/cfg.T_IMAGE_SIZE_W - 0.5
        y = pose[1]/cfg.T_IMAGE_SIZE_H - 0.5
        label = np.array([x,y])
        return label
