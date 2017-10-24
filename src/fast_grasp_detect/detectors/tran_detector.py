import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
from fast_grasp_detect.networks.success_net import SNet
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.configs.config import CONFIG
from utils.timer import Timer
import IPython
import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
slim = tf.contrib.slim
from fast_grasp_detect.visualizers.draw_cross_hair import DrawPrediction

class SDetector(object):

    def __init__(self,net_name):
        
        

        self.cfg = CONFIG()
        self.yc = YOLO_CONV(self.cfg,is_training = False)

        self.classes = self.cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = self.cfg.IMAGE_SIZE
        self.cell_size = self.cfg.CELL_SIZE
        self.boxes_per_cell = self.cfg.BOXES_PER_CELL
        self.threshold = self.cfg.THRESHOLD
        self.iou_threshold = self.cfg.IOU_THRESHOLD


        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.yc.load_network()
        self.count = 0
        self.dp = DrawPrediction()


        self.net_name = net_name

        #self.all_data = self.precompute_features(images)

        self.load_trained_net()

        #self.images_detectors()

       

    def load_trained_net(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.net = SNet(self.cfg,is_training = False)
        trained_model_file = self.cfg.TRAN_OUTPUT_DIR+ self.net_name
        print 'Restoring weights from: ' + trained_model_file
        self.variable_to_restore = slim.get_variables_to_restore()
        count = 0
        for var in self.variable_to_restore:
            print str(count) + " "+ var.name
            count += 1
        
        self.variables_to_restore = self.variable_to_restore[40:]
        self.saver_f = tf.train.Saver(self.variables_to_restore, max_to_keep=None)

        self.saver_f.restore(self.sess, trained_model_file)

        
    def precompute_features(self,images):

        all_data = []
        for image in images:

            features = self.yc.extract_conv_features(image)

            data = {}
            data['image'] = image
            data['features'] = features
            all_data.append(data)

        return all_data



    def detect(self,inputs,image):
        img_h, img_w, _ = image.shape
    
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        #IPython.embed()
       
        return net_output

   


    def predict(self,image):
       

        features = self.yc.extract_conv_features(image)
   
        result = self.detect(features,image)


        return result

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

   
    weight_file = "/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/output07_20_09_37_29save.ckpt-15000"
    #weight_file = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/save.ckpt-100'#os.path.join(args.data_dir, args.weight_dir, args.weights)
    #weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    imageList = glob.glob(os.path.join(self.cfg.IMAGE_PATH, '*.png'))
    labelListOrig = glob.glob(os.path.join(self.cfg.LABEL_PATH, '*.p'))
    labelList = [os.path.split(label)[-1].split('.')[0] for label in labelListOrig]
    #remove labeled images
    #imageList = [img for img in imageList if os.path.split(img)[-1].split('.')[0] not in labelList]

    #imname = self.cfg.IMAGE_PATH + 'frame_1000.png'
    #imname = 'test/person.jpg' 
    #IPython.embed()
    images = []
    c = 0

    detector = Detector(weight_file,images)
    for imname in imageList:

        detector.image_detector(imname)
    

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from held out image file

    


if __name__ == '__main__':
    main()
