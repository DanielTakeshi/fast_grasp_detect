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
        data: A `data_manager`, which is what takes in the original raw images (from cameras) and
            provides features to the task-specific `net`. Note, it has a `yc` argument (actually, a
            TF placeholder) that we can use to refer to the ORIGINAL input to the YOLO net, and NOT
            the pre-trained features.  For those, use `net.images`.
        """
        self.cfg = cfg
        self.net = net
        self.data = data
        self.layer = layer
        self.weights_file = self.cfg.WEIGHTS_FILE
        self.max_iter = self.cfg.MAX_ITER
        self.initial_learning_rate = self.cfg.LEARNING_RATE
        self.decay_steps = self.cfg.DECAY_STEPS
        self.decay_rate = self.cfg.DECAY_RATE
        self.staircase = self.cfg.STAIRCASE
        self.summary_iter = self.cfg.SUMMARY_ITER
        self.test_iter = self.cfg.TEST_ITER
        self.save_iter = self.cfg.SAVE_ITER
        self.original_yolo_input = data.yc.images

        # Handle output path and save config w/time-dependent string (smart!).
        self.output_dir = os.path.join(
            self.cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        # Restoring variables, if necessary.
        if not self.cfg.SMALLER_NET:
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
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grads = tf.gradients(self.net.class_loss, self.var_list)
        #for idx,item in enumerate(self.grads):
        #    print(idx,item)

        if self.cfg.FIX_PRETRAINED_LAYERS:
            self.var_list = [x for x in self.var_list if x.name in vars_restored]
        print("\ncfg.FIX_PRETRAINED_LAYERS={}. Optimizer will adjust:".format(self.cfg.FIX_PRETRAINED_LAYERS))
        numv = 0
        for item in self.var_list:
            print(item)
            numv += np.prod(item.shape)
        print("\nadjustable params: {}\n".format(numv))

        if self.cfg.OPT_ALGO == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=self.var_list)
        elif self.cfg.OPT_ALGO == 'ADAM':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=self.var_list)
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


    def variables_to_restore(self):
        return


    def train(self):
        """Called during outer script to do training!!"""
        cfg = self.cfg
        train_timer = Timer()
        load_timer = Timer()
        train_losses = []
        test_losses = []
        raw_test_losses = []  # grasp net
        raw_test_correct = [] # success net
        raw_test_total = []   # success net
        best_loss = np.float('inf')
        best_preds = None
        images_t, labels_t, c_imgs_list, d_imgs_list = self.data.get_test(return_imgs=True)

        for step in xrange(1, self.max_iter+1):
            # Get minibatch of images (usually, features from YOLO) and labels.
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            if cfg.FIX_PRETRAINED_LAYERS:
                feed_dict = {self.net.images: images, self.net.labels: labels}
            else:
                feed_dict = {self.original_yolo_input: images, self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.class_loss, self.train_op], feed_dict)
                    train_timer.toc()
                    train_losses.append(loss)

                    if step % self.test_iter == 0:
                        if cfg.FIX_PRETRAINED_LAYERS:
                            feed_dict_t = {self.net.images: images_t, self.net.labels: labels_t}
                        else:
                            feed_dict_t = {self.original_yolo_input: images_t, self.net.labels: labels_t}
                        test_loss, test_logits = self.sess.run([self.net.class_loss, self.net.logits], feed_dict_t)
                        test_losses.append(test_loss)

                        if cfg.CONFIG_NAME == 'grasp_net':
                            # Useful to get test loss in the **pixels**, not scaled version.
                            test_loss_raw = cfg.compare_preds_labels(
                                    preds=test_logits, labels=labels_t, doprint=cfg.PRINT_PREDS)
                            raw_test_losses.append(test_loss_raw)
                            print("Test loss: {:.6f} (raw: {:.2f})".format(test_loss, test_loss_raw))
                            if test_loss_raw < best_loss:
                                best_loss = test_loss_raw
                                best_preds = cfg.return_raw_labels(test_logits)
                        elif cfg.CONFIG_NAME == 'success_net':
                            correctness = np.argmax(test_logits,axis=1) == np.argmax(labels_t,axis=1)
                            correct = float(np.sum(correctness))
                            K = len(correctness)
                            raw_acc = correct / K
                            raw_test_correct.append(correct)
                            raw_test_total.append(K)
                            cfg.compare_preds_labels(preds=test_logits, labels=labels_t,
                                    correctness=correctness, doprint=cfg.PRINT_PREDS)
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

                # Don't forget best set of predictions + true labels, so we can visualize.
                cv_idx = cfg.CV_HELD_OUT_INDEX
                if cfg.CONFIG_NAME == 'grasp_net':
                    if cfg.PERFORM_CV:
                        loss_dict["cv_indices"] = cfg.CV_GROUPS[cv_idx]
                    loss_dict["preds"] = best_preds
                    loss_dict["targs"] = cfg.return_raw_labels(labels_t)

                # Save for plotting later. It should overwrite the older files saved. Careful,
                # move to another directory ASAP; e.g. if I switch datasets these overwrite.
                lrate = round(self.learning_rate.eval(session=self.sess), 6)
                img_type = 'rgb'
                if cfg.USE_DEPTH:
                    img_type = 'depth'
                suffix = '{}_type_{}_optim_{}_fixed26_{}_lrate_{}_cv_{}.p'.format(
                        cfg.CONFIG_NAME, img_type, (cfg.OPT_ALGO).lower(),
                        cfg.FIX_PRETRAINED_LAYERS, lrate, cv_idx)
                name = os.path.join(cfg.STAT_DIR, suffix)
                pickle.dump(loss_dict, open(name, 'wb'))

        # Take most recent `name` and add images here. Test set imgs should all be in order.
        name = name.replace('.p', '_raw_imgs.p')
        imgs = {'c_imgs_list':c_imgs_list, 'd_imgs_list':d_imgs_list}
        pickle.dump(imgs, open(name, 'wb'))


    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            self.cfg_dict = self.cfg.__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)
