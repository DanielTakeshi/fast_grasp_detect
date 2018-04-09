# fast_grasp_detect

## install 
python setup.py develop

## main
Contains scripts that can be run to train the networks.
'train_bed_grasp.py' - trains the grasping network
'train_bed_success.py' - trains the network that identifies successful rollouts


## src
	• CONFIG
		○ Bed_grasp_config = grasp network (trains the hsr to use end-effector to make an effective grasp)
		○ Bed_success_config = success network (tells you whether the sequence of states and actions resulted in a successfully made bed)
		○ tl_experiments/ = contains training scripts for transfer learning experiments
    ○ Other config files are for different bed-making experiments.
	• CORE
		○ Data_manager loads the yolo network and can be used to get the data as an array of images and labels
		○ Grasp_data.py is a data_manager for only the management of the grasp-net data
		○ Timer.py has 3 methods. Tic records the start time, toc returns the difference from the start time or the average time. Remain returns the time at which the timer runs out
		○ Train_fast.py trains the grasp network fast using a Solver class that takes in a network (yolo) and the dataset (pascal)
		○ Yolo_conv_features_cs.py makes the yolo CNN (builds or restores from a given weights file)
			§ Also extracts features from an image using a tensorflow feature extraction function
		○ Train_network.py = used to train the yolo network
	• DATA_AUG
		○ Augment_lighting has functions to add lighting noise to images that will later be used as training inputs
		○ Data_augment.py goes through the images and adds lighting-noisy and flipped images to the data set
	• DETECTORS
		○ Grasp_detector.py = Detects the optimal grasp point given an image of the bed
			§ Returns that grasp as a pose, which is a set of x,y coordinates that can be manipulated as necessary
		○ Tran_detector.py = detects whether the image of the bed is ready for a transition to the next robot position
			§ Detects if the robot has successfully done all it can from a spot on the floor
	• LABELERS
		○ Labeler.py = puts bounding boxes on an image upon loading it [offline]
		○ Online_labeler.py = recomputes bounding boxes after an action is completed
	• NETWORKS
		○ Grasp_net_cs.py = builds the entire grasp net
		○ Success_net.py = builds the success net (whether the bed is made or not)
	• VISUALIZERS
		○ Draw_cross_hair.py = used to draw an x on top of some location on the given image
