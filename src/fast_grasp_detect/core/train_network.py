import tensorflow as tf
import os, sys, datetime, time
from fast_grasp_detect.core.timer import Timer
import cPickle as pickle
import numpy as np
np.set_printoptions(suppress=True, precision=6)
slim = tf.contrib.slim


class Solver(object):

    def __init__(self, cfg, net, data, ss=None, layer=0):
        """
        cfg: A configuration file, have separate ones for the two tasks.
        net: Either `GHNet` or `SNet` depending on which task we're doing.
        data: A `data_manager`, which is what takes in the original raw images (from
            cameras) and provides features to the task-specific `net`. Note, it has a
            `yc` argument (actually, a TF placeholder) that we can use to refer to the
            ORIGINAL input to the YOLO net, and NOT the pre-trained features. For
            those, use `net.images`.
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
        self.original_yolo_input = data.yc.images

        # Handle output directory, (if necessary) restore variables, and get savers, etc.
        self.prepare_output_directory()
        if not cfg.SMALLER_NET:
            self.restore()
        self.all_saver  = tf.train.Saver(max_to_keep=None)
        self.summary_op = tf.summary.merge_all()

        # Support a decaying learning rate, hence use a TF placeholder.
        self.global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, self.global_step, self.decay_steps,
                self.decay_rate, self.staircase, name='learning_rate')

        # Loss function from `self.net`. Also, the variable list we want to optimize over.
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grads = tf.gradients(self.net.class_loss, self.var_list)

        if cfg.FIX_PRETRAINED_LAYERS:
            self.var_list = [x for x in self.var_list if x.name in self.vars_restored]
        print("\ncfg.FIX_PRETRAINED_LAYERS={}. Optimizer will adjust:".format(
                cfg.FIX_PRETRAINED_LAYERS))
        numv = 0
        for item in self.var_list:
            print(item)
            numv += np.prod(item.shape)
        print("\nadjustable params: {}\n".format(numv))

        if cfg.OPT_ALGO == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=self.var_list)
        elif cfg.OPT_ALGO == 'ADAM':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.net.class_loss,
                    global_step=self.global_step,
                    var_list=self.var_list)
        else:
            raise ValueError(cfg.OPT_ALGO)

        # Define the training operation and set all params via a moving average.
        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        if cfg.USE_EXP_MOV_AVG:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            self.averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.averages_op)
        else:
            self.train_op = self.optimizer

        # Use a user-specified fraction of our GPU.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEM_FRAC)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())


    def train(self):
        """The training run.
        """
        cfg = self.cfg
        train_timer = Timer()
        load_timer = Timer()
        train_losses = []
        test_losses = []
        learning_rates = []
        raw_test_losses = []  # grasp net
        raw_test_correct = [] # success net
        raw_test_total = []   # success net
        best_loss = np.float('inf')
        best_preds = None
        best_snet_acc = -1
        best_snet_preds = None
        best_snet_correctness = None
        if cfg.HAVE_TEST_SET:
            images_t, labels_t, c_imgs_list, d_imgs_list = self.data.get_test(return_imgs=True)
            test_data_sources = self.data.get_data_sources()
        elapsed_time = []
        start_t = time.time()

        for step in xrange(1, self.max_iter+1):
            # Get minibatch of images (can be features from pre-trained YOLO) and labels.
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()

            feed_dict = {self.net.labels: labels, self.net.training_mode: True}
            if cfg.FIX_PRETRAINED_LAYERS:
                feed_dict[self.net.images] = images
            else:
                feed_dict[self.original_yolo_input] = images

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.class_loss, self.train_op], feed_dict)
                    train_timer.toc()
                    train_losses.append(loss)
                    learning_rates.append(self.learning_rate.eval(session=self.sess))

                    # Normally we set `test_iter=1` so we always test when doing a summary...
                    if (step % self.test_iter == 0) and cfg.HAVE_TEST_SET:
                        feed_dict_t = {self.net.labels: labels_t, self.net.training_mode: False}
                        if cfg.FIX_PRETRAINED_LAYERS:
                            feed_dict_t[self.net.images] = images_t
                        else:
                            feed_dict_t[self.original_yolo_input] = images_t
                        test_loss, test_logits = self.sess.run(
                                [self.net.class_loss, self.net.logits], feed_dict_t)
                        test_losses.append(test_loss)

                        if cfg.CONFIG_NAME == 'grasp':
                            # Useful to get test loss in the **pixels**, not scaled version.
                            test_loss_raw = cfg.compare_preds_labels(
                                    preds=test_logits, labels=labels_t, doprint=cfg.PRINT_PREDS)
                            raw_test_losses.append(test_loss_raw)
                            print("Test loss: {:.6f} (raw: {:.2f})".format(test_loss, test_loss_raw))
                            if test_loss_raw < best_loss:
                                best_loss = test_loss_raw
                                best_preds = cfg.return_raw_labels(test_logits)
                        elif cfg.CONFIG_NAME == 'success':
                            correctness = np.argmax(test_logits,1) == np.argmax(labels_t,1)
                            correct = float(np.sum(correctness))
                            K = len(correctness)
                            raw_acc = correct / K
                            raw_test_correct.append(correct)
                            raw_test_total.append(K)
                            cfg.compare_preds_labels(preds=test_logits, labels=labels_t,
                                    correctness=correctness, doprint=cfg.PRINT_PREDS)
                            print("Test loss: {:.6f}, acc: {}/{} = {:.2f}".format(
                                    test_loss, correct, K, raw_acc))
                            if raw_acc > best_snet_acc:
                                best_snet_acc = raw_acc
                                best_snet_preds = test_logits
                                best_snet_correctness = correctness
                        else:
                            raise ValueError(self.cfg.CONFIG_NAME)

                    # Report _training_-based statistics (_not_ test-based).
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
                    summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict)
                    train_timer.toc()
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict)
                train_timer.toc()

            # Save the actual model using standard `tf.Saver`s, w/global steps. Also record
            # train/test losses. For now, only save if we're not doing cross validation.
            if step % cfg.SAVE_ITER  == 0:
                if not cfg.PERFORM_CV:
                    curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
                    ckpt_name = curr_time+ "_save.ckpt"
                    real_ckpt = os.path.join(self.HEAD_DIR, ckpt_name)
                    self.all_saver.save(self.sess, real_ckpt, global_step=self.global_step)
                    print("    saved TF model: {}".format(real_ckpt))

                # New dictionary with lists of historical info, and save (w/overwriting).
                info = {}
                if cfg.HAVE_TEST_SET:
                    info["test"] = test_losses
                    info["raw_test"] = raw_test_losses
                    info["success_test_correct"] = raw_test_correct
                    info["success_test_total"] = raw_test_total
                info["train"] = train_losses
                info["name"] = cfg.CONFIG_NAME
                info["epoch"] = self.data.epoch
                info["lrates"] = learning_rates
                elapsed_time.append( time.time()-start_t )
                info["elapsed_time"] = elapsed_time

                # Don't forget best set of predictions + true labels, so we can visualize.
                if cfg.PERFORM_CV:
                    info["cv_indices"] = cfg.CV_GROUPS[ cfg.CV_HELD_OUT_INDEX ]
                if cfg.HAVE_TEST_SET:
                    if 'grasp' in cfg.CONFIG_NAME:
                        info["preds"] = best_preds
                        info["targs"] = cfg.return_raw_labels(labels_t)
                    elif 'success' in cfg.CONFIG_NAME:
                        info["best_snet_acc"] = best_snet_acc
                        info["best_snet_preds"] = best_snet_preds
                        info["best_snet_correctness"] = best_snet_correctness
                    if len(test_data_sources) > 0:
                        info["test_data_sources"] = test_data_sources
                pickle.dump(info, open(self.stats_pickle_file, 'wb'))

        # Add test-set images (should be in order) so we can visualize later.
        if cfg.HAVE_TEST_SET:
            name = (self.stats_pickle_file).replace('.p', '_raw_imgs.p')
            imgs = {'c_imgs_list':c_imgs_list, 'd_imgs_list':d_imgs_list}
            pickle.dump(imgs, open(name, 'wb'))


    def restore(self):
        """Should only be called if we're not using smaller network.
        """
        assert not self.cfg.SMALLER_NET
        self.variable_to_restore = slim.get_variables_to_restore()
        self.variables_to_restore = self.variable_to_restore[42:52]
        count = 0
        print("\nSolver.__init__(), self.variables_to_restore:")
        self.vars_restored = []
        for var in self.variables_to_restore:
            print("{} {}".format(count, var.name))
            self.vars_restored.append(var.name)
            count += 1
        self.saver = tf.train.Saver(self.variables_to_restore, max_to_keep=None)


    def save_cfg(self, config_path):
        """Saves the `config.txt` which provides info from the (python) config file.
        """
        with open(config_path, 'w') as f:
            self.cfg_dict = self.cfg.__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)


    def prepare_output_directory(self):
        """Done _before_ training. See README of IL_ROS_HSR for high-level comments.
        """
        cfg = self.cfg
        img_type = 'rgb'
        if cfg.USE_DEPTH:
            img_type = 'depth'

        # Goes in `/.../grasp/` or `/.../success/`.
        directory = '{}_{}_img_{}_opt_{}_lr_{}_L2_{}_kp_{}_steps_{}_cv_{}'.format(
                cfg.CONFIG_NAME,
                cfg.NET_TYPE,
                img_type,
                (cfg.OPT_ALGO).lower(),
                self.initial_learning_rate,
                cfg.L2_LAMBDA,
                cfg.DROPOUT_KEEP_PROB,
                cfg.MAX_ITER,
                cfg.PERFORM_CV)

        # Also prefix this with the dataset _name_, e.g., `rollouts_white_v01`
        data_name = ( (cfg.ROLLOUT_PATH.rstrip('/')).split('/') )[-1] 

        # Now, e.g., `/.../grasp/data_name/directory`.
        self.HEAD_DIR = os.path.join(cfg.OUT_DIR, data_name, directory)
        if not os.path.exists(self.HEAD_DIR):
            os.makedirs(self.HEAD_DIR)

        # Use the stats pickle file for saving results from training.
        if cfg.PERFORM_CV:
            cidx = cfg.CV_HELD_OUT_INDEX
            self.stats_pickle_file = os.path.join(self.HEAD_DIR, 'stats_{}.p'.format(cidx))
        else:
            self.stats_pickle_file = os.path.join(self.HEAD_DIR, 'stats.p')

        # Also, for overall config, let's augment it with the exact time.
        c_name = 'config_{}.txt'.format( datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') )
        self.save_cfg( os.path.join(self.HEAD_DIR, c_name) )

