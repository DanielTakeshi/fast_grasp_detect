import numpy as np
import tensorflow as tf
import os, cv2, sys
slim = tf.contrib.slim
np.set_printoptions(edgeitems=4, suppress=True, linewidth=180)


class YOLO_CONV(object):
    """Create the bulk of the YOLO network for fixed, pre-trained weights.
    Then define fine-tuned weights in `grasp_net_cs.py` and `success_net.py`.
    """

    def __init__(self, options):
        self.cfg = cfg = options
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = int(cfg.IMAGE_SIZE)
        self.weights_file = os.path.join(cfg.PRE_TRAINED_DIR,'YOLO_small.ckpt')

        # Get logits from the input. The input is (448,448) for each channel.
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, cfg.ALPHA)

        # Use a user-specified fraction of our GPU.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEM_FRAC)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # In case we want to track some data statistics.
        self.img_mean = np.zeros((self.image_size, self.image_size, 3))


    def build_network(self, images, alpha, scope='yolo'):
        """Builds the YOLO net. We use the last layer (self.conv_layer) for pre-trained features.

        After the initial input, the next few arguments for `slim.conv2d` are:

            num_outputs: Integer, the number of output filters.
            kernel_size: A sequence of N positive integers specifying the spatial
                dimensions of the filters.  Can be a single integer to specify the same
                value for all spatial dimensions.
            stride: A sequence of N positive integers specifying the stride at which to
                compute output.  Can be a single integer to specify the same value for all
                spatial dimensions.  Specifying any `stride` value != 1 is incompatible
                with specifying any `rate` value != 1.
            padding: One of `"VALID"` or `"SAME"`.

        The `stride` defaults to 1, `padding` to SAME.

        Note: if using the smaller net, we move all the computation to the other nets.
        This part is only used for weights we don't want the user to modify, unless they
        really want that full YOLO stem.
        """
        with tf.variable_scope(scope):
            net = images

            if not self.cfg.SMALLER_NET:
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=leaky_relu(alpha),
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    #weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=slim.l2_regularizer(self.cfg.L2_LAMBDA)):
                    net = tf.pad(net, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
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

            # Important reference! We provide this to final grasp/success net layers.
            self.conv_layer = net
            return net


    def load_network(self):
        """Load network using slim's helpers and the standard `tf.train.Saver`.

        If we don't have anything to load (e.g., if using a smaller network) then this method
        shouldn't be called. Used for deployment and training IF we want pre-trained weights.

        `slim.get_variables_to_restore()`: returns list of TensorFlow variables that we _just_
        created from `build_network`. The names are important because we load them from a ckpt
        (`self.weights_file`) which assumes the same exact naming. So, it assumes we defined
        this class, then call this method. This matches `tf.Saver` properties, anyway.

        Earlier, Michael was fiddling around with the variable to restore in case we wanted to
        keep a subset of them, but let's just keep all of them for now.

        To debug, we can do:

            var = tf.trainable_variables()[k]
            print(var)
            print(self.sess.run(var))

        where k is some index, to see the value of the weights.

        By default the first '26 layers' actually take up 42 sopts in the list of variables.
        """
        assert not self.cfg.SMALLER_NET
        print('\nYOLO_CONV.load_network(), pre-trained weights: {}'.format(self.weights_file))
        vars_to_restore = slim.get_variables_to_restore()
        self.saver = tf.train.Saver(vars_to_restore, max_to_keep=None)
        self.saver.restore(self.sess, self.weights_file)


    def extract_conv_features(self, img):
        """Critical!! Runs original camera images through YOLO stem to get features.

        (ALL input to ANY deep network we have MUST go through this method!!)

        Then we later pass them to task-specific networks.  For inputs, WE DIVIDE BY 255 and then
        scale to get pixel values in [-1,1]. This gets called for _all_ the training and data points
        before training begins, so they are all held to the same scaling constraints. Same for
        testing, obviously (and fortunately!).

        Note: if we are FIXING the first 26 layers, then we want to feed through the fixed 26 layers
        and get features. Then our entire training data will 'start' from that point, and the test
        data will ALSO start from that point, and we must do the same during deployment to new
        images, i.e., pass through first 26.

        If not, we just return the image itself --- resized to the correct YOLO input. After this,
        we later pass it through the ENTIRE network, so update ALL weights.

        BTW, when we call cv2.resize(img, (448,448)) using a 3-channel img, it will correctly leave
        that 3-channel alone, and result in (448,448,3), whew. UPDATE: now supports (227,227).
        """
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        assert inputs.shape == (self.image_size, self.image_size, 3)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        # If we use a smaller net, ignore discussion about fix vs non-fixing YOLO networks.
        if self.cfg.SMALLER_NET:
            return inputs

        if self.cfg.FIX_PRETRAINED_LAYERS:
            net_output = self.sess.run(self.conv_layer, feed_dict={self.images: inputs})
            return net_output
        else:
            return inputs


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


if __name__ == '__main__':
    yc = YOLO_CONV()
    yc.load_network()
    img = cv2.imread('test.png')
    features = yc.extract_conv_features(img)
