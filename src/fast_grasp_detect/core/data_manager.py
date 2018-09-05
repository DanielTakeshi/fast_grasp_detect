import os, cv2, sys, copy, glob, IPython, time
from os.path import join
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
        self.rollout_path = cfg.ROLLOUT_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1
        self.gt_labels = None

        # Load data, following logic in our configuration file. Test set = held out.
        # With CV, make a length-1 list to maintain compatibility with non-CV case.
        num = len(cfg.CV_GROUPS)
        if cfg.PERFORM_CV:
            cidx = cfg.CV_HELD_OUT_INDEX
            self.training_list = sorted(
                    [join(cfg.ROLLOUT_PATH, cfg.CV_GROUPS[c]) for c in range(num) if c != cidx]
            )
            self.held_out_list = [ join(cfg.ROLLOUT_PATH, cfg.CV_GROUPS[cidx]) ]
        else:
            # In this case, we may or may not have a held-out test set.
            self.training_list = sorted(
                    [join(cfg.ROLLOUT_PATH, cfg.CV_GROUPS[c]) for c in range(num)]
            )
            if cfg.TEST_ROLLOUT_PATH is not None:
                self.held_out_list = sorted(
                        [join(cfg.TEST_ROLLOUT_PATH, cfg.TEST_GROUPS[c]) for c in range(num)]
                )

        # Load YOLO network and set up its pre-trained weights from known file
        # Update: we'll also use this in the case of a smaller neural network.
        print("\n`data_manager` class, now calling YOLO_CONV and loading network...")
        self.yc = YOLO_CONV(self.cfg)
        if not cfg.SMALLER_NET:
            self.yc.load_network()

        # Load test & training. Make test batch fixed (i.e, don't keep shuffling).
        self.recent_batch = []
        self.test_batch_feats  = None
        self.test_batch_labels = None
        self.test_batch_c_imgs = None
        self.test_batch_d_imgs = None
        self.train_labels = []   # All training data goes here
        self.test_labels = []    # All testing data goes here
        self.test_d_sources = [] # For combined dataset sources
        if cfg.HAVE_TEST_SET:
            self.load_test_set()
        self.load_rollouts()


    def get(self, noise=False):
        """Creates and returns images/labels for training.
        
        The images are features from the pre-trained YOLO network, or at least
        processed to correct sizes.

        Note: `count` ensures we match batch size; `cursor` for indexing into data.
        It _may_ overlap w/end of the epoch.
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
            # After we've gone through the data, re-shuffle it (not necessary but OK).
            if self.cursor >= len(self.train_labels):
                np.random.shuffle(self.train_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels


    def get_data_sources(self):
        """Return data source if we have it.
        """
        return self.test_d_sources


    def get_test(self, return_imgs=False):
        """Creates and returns images/labels for testing, where the images are
        features from the pre-trained YOLO network, not (generally) raw images.

        Unlike in training, here we should just take the entire test data in one batch.
        """
        assert self.test_batch_feats is not None and self.test_batch_labels is not None
        if return_imgs:
            assert self.test_batch_c_imgs is not None and self.test_batch_d_imgs is not None
            return (self.test_batch_feats, self.test_batch_labels,
                    self.test_batch_c_imgs, self.test_batch_d_imgs)
        else:
            return (self.test_batch_feats, self.test_batch_labels)


    def prep_image(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0
        return image


    def load_test_set(self):
        """ Assigns to `self.test_labels`, each element a dict w/relevant info.

        Form the test batch at the end. It won't be large with the data we have and we
        shouldn't keep re-computing and re-shuffling since it's a test batch.
        """
        s_time = time.time()
        cfg = self.cfg
        print("\ndata_manager.load_test_set()")

        for test_list in self.held_out_list:
            with open(test_list, 'r') as f:
                data = pickle.load(f)
            print("loaded test data: {} (length {})".format(test_list, len(data)))

            # Iterate and load data. NO data augmentation! (It's the test set!)
            # We still go through the same feature extraction process.
            for idx,item in enumerate(data):
                data_pt = {}
                if cfg.USE_DEPTH:
                    data_pt['features'] = self.yc.extract_conv_features(item['d_img'])
                else:
                    data_pt['features'] = self.yc.extract_conv_features(item['c_img'])
                data_pt['c_img'] = item['c_img']
                data_pt['d_img'] = item['d_img']
                data_pt['label'] = cfg.compute_label(item)
                self.test_labels.append(data_pt)
                if 'data_source' in item:
                    self.test_d_sources.append( item['data_source'] )

        # Form and investigate the testing images and labels in their batch.
        K = len(self.test_labels)
        print("len(self.test_labels): {}".format(K))
        if K >= 205:
            print("We'll truncate to 205 for now. (We're using one test minibatch.)")
            self.test_labels = self.test_labels[:205]
            K = 205

        # What the network needs, inputs and labels.
        self.test_batch_feats  = cfg.get_empty_state(batchdim=K)
        self.test_batch_labels = cfg.get_empty_label(batchdim=K)

        # Useful if we want to later visualize/plot test-set predictions.
        self.test_batch_c_imgs = []
        self.test_batch_d_imgs = []

        # TODO: later, if very large test set, use multiple minibatches.
        for count in range(K):
            self.test_batch_feats[count, :, :, :] = self.test_labels[count]['features']
            self.test_batch_labels[count, :]      = self.test_labels[count]['label']
            # TODO: commenting this out to save memory, plus the next debug print.
            self.test_batch_c_imgs.append( self.test_labels[count]['c_img'] )
            self.test_batch_d_imgs.append( self.test_labels[count]['d_img'] )
        print("test_batch_d_imgs[0].shape: {}".format(self.test_batch_d_imgs[0].shape))

        print("test_batch_feats.shape:     {}".format(self.test_batch_feats.shape))
        print("test_batch_labels.shape:    {}".format(self.test_batch_labels.shape))
        print("test_batch_labels:\n{}".format(self.test_batch_labels))

        if 'grasp' in cfg.CONFIG_NAME:
            W, H = cfg.T_IMAGE_SIZE_W, cfg.T_IMAGE_SIZE_H
            raw_labels = (self.test_batch_labels + 0.5) * np.array([W,H])
            print("(raw) test_batch_labels:\n{}".format(raw_labels))
        e_time = (time.time() - s_time)
        print("loaded test set in {:.3f} seconds...".format(e_time))


    def load_rollouts(self):
        """Assigns to `self.train_labels`, each element a dict w/relevant info.
        """
        s_time = time.time()
        cfg = self.cfg
        print("\ndata_manager.load_rollouts() (i.e., training data)")

        for t_list in self.training_list:
            with open(t_list, 'r') as f:
                t_list_data = pickle.load(f)
            for data in t_list_data:
                # Data augmentation magic. No need to process depth; did beforehand.
                data_a = augment_data(data, cfg.USE_DEPTH)
                for d_idx,datum_a in enumerate(data_a):
                    data_pt = {}
                    # ---------------------------------------------------------------
                    # Run YOLO network w/pre-trained weights! If we run through YOLO:
                    #       features.shape: (1, 14, 14, 1024)
                    # labels: for grasps, alternate between (x,y) and (-x,y)
                    # Note that `datum_a['img']` could represent c_img OR d_img.
                    # But when we call the network, it really 'sees' the `features`.
                    # This is where we scale pixels, resize image, etc.
                    # ---------------------------------------------------------------
                    data_pt['features'] = self.yc.extract_conv_features(datum_a['img'])
                    data_pt['label'] = cfg.compute_label(datum_a)
                    self.train_labels.append(data_pt)
            print("finished: {} (len {})".format(t_list, len(t_list_data)))

        # Labels and images/features are paired in the same dict items. :-)
        np.random.shuffle(self.train_labels)
        print("len(self.train_labels): {}. Shuffled!".format(len(self.train_labels)))
        e_time = (time.time() - s_time)
        print("loaded train set in {:.3f} seconds...".format(e_time))


    def compute_label(self, pose):
        label = np.zeros((2))
        x = float(pose[0])/cfg.T_IMAGE_SIZE_W - 0.5
        y = float(pose[1])/cfg.T_IMAGE_SIZE_H - 0.5
        label = np.array([x,y])
        return label
