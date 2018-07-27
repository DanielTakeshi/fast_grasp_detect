import numpy as np
import tensorflow as tf
import IPython
slim = tf.contrib.slim


class GHNet(object):

    def __init__(self, cfg, data_m, is_training=True, layers=0):
        """
        data_m: Needs to contain `data_m.yc.conv_layer` so that we get the tensorflow variable
            representing the 26th layer of the YOLO network, in case we need to continue passing it
            through here if not fixing the 26 layers.
        """
        self.cfg = cfg
        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE
        self.dist_size_w = self.cfg.T_IMAGE_SIZE_W/self.cfg.RESOLUTION
        self.dist_size_h = self.cfg.T_IMAGE_SIZE_H/self.cfg.RESOLUTION
        self.output_size = 2 # 2-D output because we are predicting (x,y) pixel coords
        self.layers = layers
        self.learning_rate = self.cfg.LEARNING_RATE
        self.batch_size = self.cfg.BATCH_SIZE
        self.alpha = self.cfg.ALPHA
        self.yolo_conv_layer = data_m.yc.conv_layer

        # Despite the name, `self.images` are features from the YOLO stem in `core/yolo_conv_features_cs.py`
        if cfg.FIX_PRETRAINED_LAYERS:
            if self.layers == 0:
                self.images = tf.placeholder(tf.float32, [None, self.cfg.FILTER_SIZE, self.cfg.FILTER_SIZE, self.cfg.NUM_FILTERS], name='images')
            elif self.layers == 1:
                self.images = tf.placeholder(tf.float32, [None, self.cfg.FILTER_SIZE_L1, self.cfg.FILTER_SIZE_L1, self.cfg.NUM_FILTERS], name='images')
            elif self.layers == 2:
                self.images = tf.placeholder(tf.float32, [None, self.cfg.SIZE_L2], name='images')
            # logits are not log prob of classes but simply the predicted pixels
            self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)
        else:
            self.logits = self.build_network(self.yolo_conv_layer, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            # self.labels to be provided by the human user
            self.labels = tf.placeholder(tf.float32, [None, 2])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)


    def build_network(self, images, num_outputs, alpha, keep_prob=1.0, is_training=True, scope='yolo'):
        """Extra layers built on _top_ of the YOLO stem (first 26 layers)."""
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                if self.layers == 0:
                    net = tf.pad(images, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                    net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                    net = slim.conv2d(net, 1024, 3, scope='conv_29')
                    net = slim.conv2d(net, 1024, 3, scope='conv_30')
                    net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                    net = slim.flatten(net, scope='flat_32')

                elif self.layers == 1:
                    net = slim.conv2d(images, 1024, 3, scope='conv_30')
                    net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                    net = slim.flatten(net, scope='flat_32')

                elif self.layers == 2:
                    net = slim.flatten(images, scope='flat_32')

                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net


    def loss_layer(self, predict_classes, classes, scope='loss_layer'):
        """Despite the names here, this should be standard mean square error (L2) loss."""
        with tf.variable_scope(scope):
            class_delta = (predict_classes - classes) # not `class` but just error
            self.class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1]), name='class_loss')
            tf.losses.add_loss(self.class_loss)
            tf.summary.scalar('class_loss', self.class_loss)


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
