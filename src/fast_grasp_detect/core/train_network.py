import tensorflow as tf
import datetime
import os
import sys
import argparse
from fast_grasp_detect.core.timer import Timer
import IPython
import cPickle as pickle
import numpy as np
np.set_printoptions(suppress=True, precision=6)
slim = tf.contrib.slim


class Solver(object):

    def __init__(self, cfg, net, data, ss=None, layer=0):
        """
        cfg: A configuration file, have separate ones for the two tasks.
        net: Either `GHNet` or `SNet` depending on which task we're doing.
        data: A `data_manager`, which is what takes in the original raw images
            (from cameras) and provides features to the task-specific `net`.
        """
        self.cfg = cfg
        self.net = net
        self.data = data
        self.weights_file = self.cfg.WEIGHTS_FILE
        self.max_iter = self.cfg.MAX_ITER
        self.initial_learning_rate = self.cfg.LEARNING_RATE
        self.decay_steps = self.cfg.DECAY_STEPS
        self.decay_rate = self.cfg.DECAY_RATE
        self.staircase = self.cfg.STAIRCASE
        self.summary_iter = self.cfg.SUMMARY_ITER
        self.test_iter = self.cfg.TEST_ITER
        self.viz_debug_iter = self.cfg.VIZ_DEBUG_ITER
        self.layer = layer
        self.ss = ss
        self.save_iter = self.cfg.SAVE_ITER

        # Handle output path and save config w/time-dependent string (smart!).
        self.output_dir = os.path.join(
            self.cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        # Restoring variables.
        self.variable_to_restore = slim.get_variables_to_restore()
        self.variables_to_restore = self.variable_to_restore[42:52]
        count = 0
        print("\nSolver.__init__(), self.variables_to_restore:")
        vars_restored = []
        for var in self.variables_to_restore:
            print("{} {}".format(count, var.name))
            vars_restored.append(var.name)
            count += 1
        self.saver = tf.train.Saver(self.variables_to_restore, max_to_keep=None)
        self.all_saver = tf.train.Saver()
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, self.global_step, self.decay_steps,
                self.decay_rate, self.staircase, name='learning_rate')

        # Loss function from `self.net`. Also adding the variable list we want to optimize over.
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.cfg.FIX_PRETRAINED_LAYERS:
            var_list = [x for x in var_list if x.name in vars_restored]
        print("\ncfg.FIX_PRETRAINED_LAYERS={}. Our optimizer will adjust:".format(self.cfg.FIX_PRETRAINED_LAYERS))
        for item in var_list:
            print(item)

        if self.cfg.OPT_ALGO == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=var_list)
        elif self.cfg.OPT_ALGO == 'ADAM':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=var_list)
        else:
            raise ValueError(self.cfg.OPT_ALGO)

        # Define the training operation and set all params via a moving average.
        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        if self.cfg.USE_EXP_MOV_AVG:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            self.averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.averages_op)
        else:
            self.train_op = self.optimizer

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # Daniel: by default, it's None. We've already restored earlier in the data manager class.
        if self.weights_file is not None:
            print('\n(after tf initializer) Solver.__init__(), restoring weights for net from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)
        else:
            print('\n(after tf initializer) self.weights_file is None, so not restoring here.')


    def variables_to_restore(self):
        return


    def train(self):
        """Called during outer script to do training!!"""
        train_timer = Timer()
        load_timer = Timer()
        train_losses = []
        test_losses = []
        raw_test_losses = []  # for grasp net
        raw_test_correct = [] # for success net
        raw_test_total = []   # for success net

        for step in xrange(1, self.max_iter+1):
            # Get minibatch of images (usually, features from YOLO) and labels.
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images, self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.class_loss, self.train_op], feed_dict=feed_dict)
                    train_timer.toc()
                    train_losses.append(loss)

                    if step % self.test_iter == 0:
                        images_t, labels_t = self.data.get_test()
                        feed_dict_test = {self.net.images: images_t, self.net.labels: labels_t}
                        test_loss, test_logits = self.sess.run([self.net.class_loss, self.net.logits], feed_dict=feed_dict_test)
                        test_losses.append(test_loss)

                        if self.cfg.CONFIG_NAME == 'grasp_net':
                            # Useful to get test loss in the **pixels**, not scaled version.
                            test_loss_raw = self.cfg.compare_preds_labels(
                                    preds=test_logits, labels=labels_t, doprint=True)
                            raw_test_losses.append(test_loss_raw)
                            print("Test loss: {:.6f} (raw: {:.2f})".format(test_loss, test_loss_raw))
                        elif self.cfg.CONFIG_NAME == 'success_net':
                            correctness = np.argmax(test_logits,axis=1) == np.argmax(labels_t,axis=1)
                            correct = float(np.sum(correctness))
                            K = len(correctness)
                            raw_acc = correct / K
                            raw_test_correct.append(correct)
                            raw_test_total.append(K)
                            self.cfg.compare_preds_labels(preds=test_logits, labels=labels_t,
                                    correctness=correctness, doprint=True)
                            print("Test loss: {:.6f}, acc: {}/{} = {:.2f}".format(test_loss, correct, K, raw_acc))
                        else:
                            raise ValueError(self.cfg.CONFIG_NAME)

                    log_str = ('{} Epoch: {}, Step: {}, L-Rate: {},'
                        ' Loss: {:.6f}\nSpeed: {:.3f}s/iter,'
                        ' Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)
                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            # Save the actual model using standard `tf.Saver`s, also record train/test losses.
            if step % self.save_iter == 0:
                curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
                real_out = self.cfg.OUTPUT_DIR
                real_ckpt = real_out + curr_time + "_CS_"+str(self.layer)+ "_save.ckpt"
                print("    saving tf checkpoint to {}".format(real_ckpt))
                self.all_saver.save(self.sess, real_ckpt, global_step=self.global_step)
                loss_dict = {}
                loss_dict["test"] = test_losses
                loss_dict["raw_test"] = raw_test_losses
                loss_dict["success_test_correct"] = raw_test_correct
                loss_dict["success_test_total"] = raw_test_total
                loss_dict["train"] = train_losses
                loss_dict["name"] = self.cfg.CONFIG_NAME
                loss_dict["epoch"] = self.data.epoch
                # Use this for plotting. It should overwrite the older files saved. Careful, move
                # these to another directory ASAP; e.g. if I switch datasets these overwrite.
                lrate = round(self.learning_rate.eval(session=self.sess), 6)
                suffix = '{}_depth_{}_optim_{}_fixed_{}_lrate_{}.p'.format(self.cfg.CONFIG_NAME,
                        self.cfg.USE_DEPTH, self.cfg.OPT_ALGO, self.cfg.FIX_PRETRAINED_LAYERS, lrate)
                name = os.path.join(self.cfg.STAT_DIR, suffix)
                pickle.dump(loss_dict, open(name, 'wb'))


    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            self.cfg_dict = self.cfg.__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)
