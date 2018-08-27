import os, glob, cv2, argparse, sys
from fast_grasp_detect.networks.grasp_net_cs import GHNet
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python import pywrap_tensorflow


class GDetector(object):
    """Loaded when _deploying_ the learned bed-making grasping policy."""

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

        # Now load the part _specific_ to the grasping net.
        self.load_trained_net()


    def load_trained_net(self):
        """Load in network that has been trained using slim and tf.Saver.

        A few odd notes. You can inspect variables stored in a checkpoint file.

            reader = pywrap_tensorflow.NewCheckpointReader(trained_model_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print(key)
        
        Another way to debug:

            var = tf.trainable_variables()[k]
            print(var)
            print(self.sess.run(var))

        However, YOLO_CONV and this class have their own TensorFlow sessions.
        With multiple sessions, they have different values for same-named
        variables.  Oddly, when I restore our trained model file, the first 26
        layers (if we're using fixed weights) are NOT loaded correctly.  But
        fortunately the ones after are correctly loaded and that's all we need,
        because in this class, we do not call the first 26 layers but instead
        refer to the YOLO_CONV class (and thus the YOLO_CONV's TensorFlow
        session).

        Update: only restore if 'grasp' is in name for this session.
        """
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.net = GHNet(cfg=self.fg_cfg, yc=self.yc, is_training=False)
        trained_model_file = self.bed_cfg.GRASP_NET_PATH
        vars_to_restore = [x for x in slim.get_variables_to_restore()
                if 'grasp' in x.name]

        print('\nGDetector.load_trained_net(), Restoring:\n{}'.format(
                trained_model_file))
        print("num vars to restore: {}".format(len(vars_to_restore)))
        self.saver_f = tf.train.Saver(vars_to_restore, max_to_keep=None)
        self.saver_f.restore(self.sess, trained_model_file)


    def predict(self, image, draw_cross_hair=False):
        """Called during deployment code! Maps [-1,1] prediction to raw pixels.

        As expected, we must pass it through the SAME processing code, inside
        the YOLO class and `extract_conv_features`. This will run it through the
        YOLO's TensorFlow session if we decided to use their pre-trained
        features!

        THE DEPTH IMAGES MUST BE PRE-PROCESSED BEFOREHAND!!!
        """
        features = self.yc.extract_conv_features(image)
        result = self.detect(features, image)
        x = self.fg_cfg.T_IMAGE_SIZE_W * (result[0,0] + 0.5)
        y = self.fg_cfg.T_IMAGE_SIZE_H * (result[0,1] + 0.5)
        pose = np.array([x,y])

        # Draws a large cross hair over the image. Should be handled outside.
        if draw_cross_hair:
            img = self.dp.draw_prediction(image, pose)
            cv2.imshow('detected_result',img)
            cv2.waitKey(30)
        return pose


    def detect(self, inputs, image):
        """Called (class internally only?) to run through the post-YOLO network.
        """
        img_h, img_w, _ = image.shape
        feed = {self.net.images: inputs, self.net.training_mode: False}
        net_output = self.sess.run(self.net.logits, feed)
        return net_output


if __name__ == '__main__':
    pass
