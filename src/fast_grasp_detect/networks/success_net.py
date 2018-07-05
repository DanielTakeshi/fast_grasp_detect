import numpy as np
import tensorflow as tf
#import yolo.config_card as self.cfg
import IPython
slim = tf.contrib.slim


class SNet(object):

    def __init__(self, cfg, is_training=True, layers=0):
        self.cfg = cfg
        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE
        self.dist_size_w = self.cfg.T_IMAGE_SIZE_W/self.cfg.RESOLUTION
        self.dist_size_h = self.cfg.T_IMAGE_SIZE_H/self.cfg.RESOLUTION
        self.output_size = 2
        self.learning_rate = self.cfg.LEARNING_RATE
        self.batch_size = self.cfg.BATCH_SIZE
        self.alpha = self.cfg.ALPHA
        self.layers = layers

        # Like with the grasp net, these images are features. Also, logits is
        # the output _after_ a softmax operation (so it's not log probs).
        self.images = tf.placeholder(tf.float32, [None, self.cfg.FILTER_SIZE, self.cfg.FILTER_SIZE, self.cfg.NUM_FILTERS], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, 2])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
        #self.images = tf.placeholder(tf.float32, [None, self.cfg.FILTER_SIZE, self.cfg.FILTER_SIZE, self.cfg.NUM_FILTERS], name='images')


    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=1.0,
                      is_training=True,
                      scope='yolo'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
                net = tf.nn.softmax(net) # Note the softmax here!
        return net


    def loss_layer(self, predicts, classes, scope='loss_layer'):
        """ For transitions, the loss is also L2 (actually the paper says
        absolute loss, need to fixx that ...). Should we do cross entropy?
        """
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts, [self.batch_size,2])
            class_delta = (predict_classes - classes)
            self.class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1]), name='class_loss')
            tf.losses.add_loss(self.class_loss)
            tf.summary.scalar('class_loss', self.class_loss)


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
