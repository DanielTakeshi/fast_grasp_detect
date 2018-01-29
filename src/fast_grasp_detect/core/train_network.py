import tensorflow as tf
import datetime
import os
import sys
import argparse


from fast_grasp_detect.core.timer import Timer

import IPython
import cPickle as pickle
slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Solver(object):

    def __init__(self,cfg, net, data,ss = None,layer = 0):

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
        self.output_dir = os.path.join(
            self.cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.save_cfg()

        
        self.variable_to_restore = slim.get_variables_to_restore()
        self.variables_to_restore = self.variable_to_restore[42:52]

        count = 0
        for var in self.variables_to_restore:
            print str(count) + " "+ var.name
            count += 1
        #tf.global_variables_initializer()
        self.saver = tf.train.Saver(self.variables_to_restore, max_to_keep=None)

        self.all_saver = tf.train.Saver()
        #self.saver = tf.train.Saver()
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()




        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
       
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.class_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        #IPython.embed()

        if self.weights_file is not None:
            print('Restoring weights for net from: ' + self.weights_file)
         
            self.saver.restore(self.sess, self.weights_file)

       
    def variables_to_restore(self):

        return

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        train_losses = []
        test_losses = []

        for step in xrange(1, self.max_iter + 1):
           
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images, self.net.labels: labels}

            
            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.class_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                    train_losses.append(loss)

                   

                    if(step % self.test_iter) == 0:
                        images_t, labels_t = self.data.get_test()
                     
                        feed_dict_test = {self.net.images : images_t, self.net.labels: labels_t}
                        

                        test_loss = self.sess.run(
                            self.net.class_loss,
                            feed_dict=feed_dict_test)


                        test_losses.append(test_loss)
                        print("Test loss: " + str(test_loss))
                        

                       

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
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

            if step % self.save_iter == 0:
                # # print('{} Saving checkpoint file to: {}'.format(
                #     datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                #     self.output_dir))

                curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')


                real_out = self.cfg.OUTPUT_DIR
                
                real_ckpt = real_out + curr_time + "_CS_"+str(self.layer)+ "_save.ckpt"
                print("saving to " + str(real_out))

                self.all_saver.save(self.sess, real_ckpt,
                                global_step=self.global_step)
                loss_dict = {}
                loss_dict["test"] = test_losses
                loss_dict["train"] = train_losses
                loss_dict["name"] = self.cfg.CONFIG_NAME

               
                pickle.dump(loss_dict, open(self.cfg.STAT_DIR+self.cfg.CONFIG_NAME+'.p', 'wb'))

        return min(test_losses)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            self.cfg_dict = self.cfg.__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)



    #IPython.embed()

