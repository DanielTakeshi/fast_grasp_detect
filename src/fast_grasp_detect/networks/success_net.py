import numpy as np
import tensorflow as tf
import IPython, os, sys
slim = tf.contrib.slim


class SNet(object):

    def __init__(self, cfg, yc, is_training=True):
        """
        yc: Needs to contain `yc.conv_layer` so that we get the TF variable representing
            the 26th layer of the YOLO network, in case we need to continue passing it
            through here if not fixing the 26 layers.
        is_training: NOT for dropout/batch_norm/etc reasons but for deciding whether we
            should make some new TF variables for the losses, etc. Should only be false
            if we are using this during bed-making _deployment_.
        """
        self.cfg = cfg
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = int(cfg.IMAGE_SIZE)
        self.dist_size_w = cfg.T_IMAGE_SIZE_W / cfg.RESOLUTION
        self.dist_size_h = cfg.T_IMAGE_SIZE_H / cfg.RESOLUTION
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = int(cfg.BATCH_SIZE)
        self.alpha = cfg.ALPHA
        self.yolo_conv_layer = yc.conv_layer

        # 2D output where [1,0] means 100 % probability for class 0 (success), etc.
        self.output_size = 2

        # In case we want to apply dropout. Training only! Currently confusing since there is
        # another `is_training` from earlier. This here helps to see _validation_ performance.
        self.training_mode = tf.placeholder(tf.bool, name='training_mode')
        self.keep_prob = cfg.DROPOUT_KEEP_PROB

        # Despite the name, `images` could be features from YOLO stem. Also, `logits` may
        # be the output before or after a softmax op depending on the loss.
        if cfg.FIX_PRETRAINED_LAYERS:
            assert not cfg.SMALLER_NET
            fs = int(cfg.FILTER_SIZE)
            nf = int(cfg.NUM_FILTERS)
            self.images = tf.placeholder(tf.float32, [None, fs, fs, nf], name='images')
            self.logits = self.build_network(self.images, self.output_size, self.alpha, self.training_mode)
        else:
            self.logits = self.build_network(self.yolo_conv_layer, self.output_size, self.alpha, self.training_mode)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, 2])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)


    def build_network(self, images, num_outputs, alpha, training_mode):
        """Extra layers built on _top_ of the YOLO stem (first 26 layers).
        
        Try Xavier init for end-to-end training, truncated normal for YOLO?
        """
        with tf.variable_scope('success'):
            net = images

            if self.cfg.NET_TYPE == 3 or self.cfg.NET_TYPE == 4:
                pad = 2
                if (self.cfg.NET_TYPE == 4):
                    pad = 1
                assert self.cfg.SMALLER_NET

                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    #weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=slim.l2_regularizer(self.cfg.L2_LAMBDA)):
                    net = slim.conv2d(net, 64, [7, 7], pad, padding='SAME')
                    net = slim.conv2d(net, 128, [5, 5], 2, padding='SAME')
                    net = slim.max_pool2d(net, [2, 2], 2)

                    net = slim.conv2d(net, 128, [5, 5])
                    net = slim.conv2d(net, 192, [3, 3], 2)
                    net = slim.max_pool2d(net, [2, 2], 2)

                    net = slim.conv2d(net, 192, [3, 3])
                    net = slim.conv2d(net, 192, [3, 3])
                    net = slim.conv2d(net, 128, [3, 3])
                    net = slim.max_pool2d(net, [2, 2], 2)

                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 2000)
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=training_mode)
                    net = slim.fully_connected(net, 2000)
                    net = slim.fully_connected(net, num_outputs, activation_fn=None)

            else:
                assert not self.cfg.SMALLER_NET
                assert self.cfg.NET_TYPE == 1 or self.cfg.NET_TYPE == 2
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=leaky_relu(alpha),
                                    weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                                    #weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=slim.l2_regularizer(self.cfg.L2_LAMBDA)):
                    net = slim.conv2d(net, 256, 3, stride=2, scope='conv_29')
                    net = slim.conv2d(net, 256, 3, scope='conv_30')
                    net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                    net = slim.flatten(net, scope='flat_32')

                    # The YOLO paper only did a dropout after the first FC layer.
                    net = slim.fully_connected(net, 1024, scope='fc_33')
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=training_mode)
                    net = slim.fully_connected(net, 1024, scope='fc_34')
                    net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        #get_variables()
        return net


    def loss_layer(self, predict_classes, classes, scope='loss_layer'):
        """For transitions, the loss is CE or soft-L2

        `predict_classes` are logits for former, softmax(logits) for latter.
        """
        with tf.variable_scope(scope):
            if self.cfg.CROSS_ENT_LOSS:
                self.class_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                            logits=predict_classes, labels=classes),
                    name='class_loss'
                )
            else:
                class_delta = (predict_classes - classes)
                self.class_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(class_delta), axis=[1]), name='class_loss'
                )
            tf.losses.add_loss(self.class_loss)
            tf.summary.scalar('class_loss', self.class_loss)


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op


def get_variables():
    print("")
    variables = tf.trainable_variables()
    numv = 0
    for vv in variables:
        numv += np.prod(vv.shape)
        print(vv)
    print("\nNumber of parameters: {}".format(numv))
