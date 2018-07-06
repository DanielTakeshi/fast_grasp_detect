import numpy as np
import tensorflow as tf
import IPython
import os
import cv2
slim = tf.contrib.slim


class YOLO_CONV(object):
    """Create the bulk of the YOLO network for fixed, pre-trained weights.
    Then define fine-tuned weights in `grasp_net_cs.py` and `success_net.py`.
    """

    def __init__(self, options, is_training=True, layer=0):
        self.cfg = options
        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE
        self.cell_size = self.cfg.CELL_SIZE
        self.boxes_per_cell = self.cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        self.alpha = self.cfg.ALPHA
        self.layers = layer

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        # Get logits from the input, which are (448,448) for each channel.
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):

        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                self.conv_layer = net

                # Add more layers if desired, but by default just keep first 26.
                if self.layers >= 1:
                    net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                    net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                    net = slim.conv2d(net, 1024, 3, scope='conv_29')
                    self.conv_layer = net

                if self.layers >= 2:
                    net = slim.conv2d(net, 1024, 3, scope='conv_30')
                    net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                    net = slim.flatten(net, scope='flat_32')
                    self.conv_layer = net

                return net


    def load_network(self):
        """Load network using slim's helpers and the standard `tf.train.Saver`."""
        self.weights_file = self.cfg.PRE_TRAINED_DIR+"YOLO_small.ckpt"
        print('\nIn YOLO_CONV.load_network(), restoring weights from: {}'.format(self.weights_file))
        self.variable_to_restore = slim.get_variables_to_restore()
        count = 0
        for var in self.variable_to_restore:
            print str(count) + " "+ var.name
            count += 1

        if self.layers == 0:
            print("self.layers = 0 so self.variables_to_restore: 0:45")
            self.variables_to_restore = self.variable_to_restore[0:45]
        elif self.layers == 1:
            print("self.layers = 1 so self.variables_to_restore: 0:46")
            self.variables_to_restore = self.variable_to_restore[0:46]
        elif self.layers == 2:
            print("self.layers = 2 so self.variables_to_restore: 0:48")
            self.variables_to_restore = self.variable_to_restore[0:48]

        #tf.global_variables_initializer()
        self.saver = tf.train.Saver(self.variables_to_restore, max_to_keep=None)
        #self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)


    def extract_conv_features(self, img):
        """Critical!! Runs original camera images through YOLO stem to get
        features. Then we later pass them to task-specific networks.
        """
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        #inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))
        net_output = self.sess.run(self.conv_layer, feed_dict={self.images: inputs})
        return net_output


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op


if __name__ == '__main__':
    yc = YOLO_CONV()
    yc.load_network()
    img = cv2.imread('test.png')
    features = yc.extract_conv_features(img)
