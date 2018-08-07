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
        self.cfg = cfg = options
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

        # For cross-validation. Use self.training_list to hold single list of all training
        # rollouts. That way I can ignore rollouts in the same target directory simply by
        # removing the index in the configuration files rather than moving the rollout file.
        if cfg.PERFORM_CV:
            cidx = cfg.CV_HELD_OUT_INDEX
            num = len(cfg.CV_GROUPS)
            self.held_out_list = cfg.CV_GROUPS[cidx]
            self.training_list = sorted(
                    sum([cfg.CV_GROUPS[c] for c in range(num) if c != cidx], [])
            )
            self.held_out_rollouts = [os.path.join(self.rollout_path,'rollout_'+str(rr)) 
                    for rr in self.held_out_list]
            self.training_rollouts = [os.path.join(self.rollout_path,'rollout_'+str(rr)) 
                    for rr in self.training_list]

        # Load YOLO network and set up its pre-trained weights from known file
        # Update: we'll also use this in the case of a smaller neural network.
        print("\n`data_manager` class, now calling YOLO_CONV and loading network...")
        self.yc = YOLO_CONV(self.cfg)
        if not self.cfg.SMALLER_NET:
            self.yc.load_network()

        # Load test & training rollouts. Make test batch fixed (i.e, don't keep shuffling).
        self.recent_batch = []
        self.test_batch_feats  = None
        self.test_batch_labels = None
        self.test_batch_c_imgs = None
        self.test_batch_d_imgs = None
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


    def get_test(self, return_imgs=False):
        """Creates and returns images/labels for testing, where the images are
        features from the pre-trained YOLO network, not (generally) raw images.

        Unlike in training, here we should just take the entire test data in one batch.
        """
        assert self.test_batch_feats is not None and self.test_batch_labels is not None
        if return_imgs:
            assert self.test_batch_c_imgs is not None and self.test_batch_d_imgs is not None
            return (self.test_batch_feats, self.test_batch_labels, self.test_batch_c_imgs, self.test_batch_d_imgs)
        else:
            return (self.test_batch_feats, self.test_batch_labels)


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
            print("\ndata_manager.load_test_set(), held-out: {} (cv index {}) from {}".format(
                    self.held_out_list, self.cfg.CV_HELD_OUT_INDEX, self.rollout_path))
            rollouts = list(self.held_out_rollouts)
        else:
            print("\ndata_manager.load_test_set(): {}".format(self.cfg.BC_HELD_OUT))
            rollouts = glob.glob(os.path.join(self.cfg.BC_HELD_OUT, '*_*'))

        self.test_labels = []

        # Oops, bad naming, `grasp_rollout` could also contain the success, but w/e.
        for rollout_p in rollouts:
            rollout = pickle.load(open(rollout_p+'/rollout.p'))
            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            print("{},  len(relevant)={},  w/len(rollout)={} [TEST]".format(
                    rollout_p, len(grasp_rollout), len(rollout)))

            for grasp_point in grasp_rollout:
                data_pt = {}
                # Run the YOLO network w/pre-trained weights!!
                # No data augmentation, since we're on the test set.
                # Also the datum_to_net_dim only changes 'd_img'.
                grasp_point[0] = datum_to_net_dim(grasp_point[0])
                if self.cfg.USE_DEPTH:
                    data_pt['features'] = self.yc.extract_conv_features(grasp_point[0]['d_img'])
                else:
                    data_pt['features'] = self.yc.extract_conv_features(grasp_point[0]['c_img'])
                data_pt['c_img'] = grasp_point[0]['c_img']
                data_pt['d_img'] = grasp_point[0]['d_img']
                data_pt['label'] = self.cfg.compute_label(grasp_point[0])
                self.test_labels.append(data_pt)

        # Form and investigate the testing images and labels in their batch.
        K = len(self.test_labels)
        print("len(self.test_labels): {}".format(K))
        assert K <= 500

        # What the network needs, inputs and labels.
        self.test_batch_feats = self.cfg.get_empty_state(batchdim=K)
        self.test_batch_labels = self.cfg.get_empty_label(batchdim=K)

        # Useful if we want to later visualize/plot test-set predictions.
        self.test_batch_c_imgs = []
        self.test_batch_d_imgs = []

        for count in range(K):
            self.test_batch_feats[count, :, :, :] = self.test_labels[count]['features']
            self.test_batch_labels[count, :]      = self.test_labels[count]['label']
            self.test_batch_c_imgs.append( self.test_labels[count]['c_img'] )
            self.test_batch_d_imgs.append( self.test_labels[count]['d_img'] )

        print("test_batch_d_imgs[0].shape: {}".format(self.test_batch_d_imgs[0].shape))
        print("test_batch_feats.shape:     {}".format(self.test_batch_feats.shape))
        print("test_batch_labels.shape:    {}".format(self.test_batch_labels.shape))
        print("test_batch_labels:\n{}".format(self.test_batch_labels))

        if self.cfg.CONFIG_NAME == 'grasp_net':
            W, H = self.cfg.T_IMAGE_SIZE_W, self.cfg.T_IMAGE_SIZE_H
            raw_labels = (self.test_batch_labels + 0.5) * np.array([W,H])
            print("(raw) test_batch_labels:\n{}".format(raw_labels))


    def load_rollouts(self):
        """ Assigns to `self.train_labels`, each element a dict w/relevant info.

        Each rollout in the designated directory is broken up into 'grasp_point' dicts, which are
        like the data dicts we used for collecting data (though w/out 'd_img'). Then for each, we
        augment the data and then run the YOLO network on that to get pre-extracted features. Note
        that these features are assumed to be held fixed, so if we did this in test-time execution,
        we would need those test images passed through this same feature extractor, THEN through our
        pre-trained weights.
        
        Unless we trained the entire YOLO net, that is, in which case the feature extractor is
        simply something that resizes the input for the first YOLO layer.
        """
        if self.cfg.PERFORM_CV:
            print("\ndata_manager.load_rollouts(), {} (held-out ignored: {}, idx {})".format(
                    self.rollout_path, self.held_out_list, self.cfg.CV_HELD_OUT_INDEX))
            rollouts = list(self.training_rollouts)
        else:
            print("\ndata_manager.load_rollouts(), path: {}".format(self.rollout_path))
            rollouts = glob.glob(os.path.join(self.rollout_path, '*_*'))

        self.train_labels = []

        # Oops, bad naming, `grasp_rollout` could also contain the success, but w/e.
        for rollout_p in rollouts:
            rollout = pickle.load(open(rollout_p+'/rollout.p'))
            grasp_rollout = self.cfg.break_up_rollouts(rollout)
            print("{},  len(relevant)={},  w/len(rollout)={}".format(
                    rollout_p, len(grasp_rollout), len(rollout)))

            for grasp_point in grasp_rollout:
                for data in grasp_point:
                    c_img = data['c_img']
                    d_img = data['d_img']
                    data = datum_to_net_dim(data) # always do _before_ data augmentation!!
                    data_a = augment_data(data, self.cfg.USE_DEPTH) # data augmentation magic.
                    for d_idx,datum_a in enumerate(data_a):
                        data_pt = {}
                        # Run the YOLO network w/pre-trained weights! 
                        # features.shape: (1, 14, 14, 1024)
                        # labels: for grasps, alternate between (x,y) and (-x,y)
                        # Note that `datum_a['img']` could represent c_img OR d_img.
                        # But when we call the network, it really 'sees' the `features`.
                        data_pt['features'] = self.yc.extract_conv_features(datum_a['img'])
                        data_pt['label'] = self.cfg.compute_label(datum_a)
                        #data_pt['c_img'] = c_img
                        #data_pt['d_img'] = d_img
                        self.train_labels.append(data_pt)

        np.random.shuffle(self.train_labels)
        print("len(self.train_labels): {}. Also, shuffled!".format(len(self.train_labels)))


    def compute_label(self, pose):
        """Load image and bounding boxes info from XML file in the PASCAL VOC format."""
        label = np.zeros((2))
        x = float(pose[0])/cfg.T_IMAGE_SIZE_W - 0.5
        y = float(pose[1])/cfg.T_IMAGE_SIZE_H - 0.5
        label = np.array([x,y])
        return label
