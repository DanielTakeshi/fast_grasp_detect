import os, glob, cv2, argparse, sys
from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from tensorflow.python import pywrap_tensorflow


class SDetector(object):
    """Loaded when _deploying_ the learned success/transition policy."""

    def __init__(self, fg_cfg, bed_cfg, yc=None):
        """To load this, we utilize two configuration files.

        fg_cfg: use for training, in fast_grasp_detect
        bed_cfg: use for bed-making now, for collection or deployment
        """
        self.fg_cfg = fg_cfg
        self.bed_cfg = bed_cfg
        self.classes = fg_cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = int(fg_cfg.IMAGE_SIZE)
        self.dp = DrawPrediction()

        # If yc is NOT none, then we've already built + loaded elsewhere.
        if yc is None:
            self.yc = YOLO_CONV(fg_cfg)
            self.yc.load_network()
        else:
            self.yc = yc

        # Now load the part _specific_ to the success net.
        self.load_trained_net()
       

    def load_trained_net(self):
        """Similar to the corresponding method in `GDetector` class.
        
        Only restoring if 'success' is in the name.
        """
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.net = SNet(cfg=self.fg_cfg, yc=self.yc, is_training=False)
        trained_model_file = self.bed_cfg.SUCC_NET_PATH
        vars_to_restore = [x for x in slim.get_variables_to_restore()
                if 'success' in x.name]

        print('\nSDetector.load_trained_net(), Restoring:\n{}'.format(
                trained_model_file))
        print("num vars to restore: {}".format(len(vars_to_restore)))
        self.saver_f = tf.train.Saver(vars_to_restore, max_to_keep=None)
        self.saver_f.restore(self.sess, trained_model_file)


    def predict(self, image):
        """Call during deployment code. Similar to GDetector's method.
        
        THE DEPTH IMAGES MUST BE PRE-PROCESSED BEFOREHAND!!!
        """
        features = self.yc.extract_conv_features(image)
        result = self.detect(features, image)
        return result


    def detect(self, inputs, image):
        """Called internally only."""
        img_h, img_w, _ = image.shape
        feed = {self.net.images: inputs, self.net.training_mode: False}
        net_output = self.sess.run(self.net.logits, feed)
        return net_output

    
if __name__ == '__main__':
    pass
