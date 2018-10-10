# fast_grasp_detect

This is a snapshot of the code used for:

- Robot Bed-Making: Deep Transfer Learning Using Depth Sensing of Deformable Fabric
- Daniel Seita, Nawid Jamali, Michael Laskey, Ron Berenstein, Ajay Kumar Tanwani, Prakash Baskaran,
  Soshi Iba, John Canny, Ken Goldberg
- https://arxiv.org/abs/1809.09810

**This repository will not be updated further, to ensure that code can be re-run as needed to
reproduce results**. To see any follow-up work, check out the BerkeleyAutomation version of this
repository (use a separate virtualenv).

## install 

```
python setup.py develop
```

(Or `pip install -e .` in this directory ... that might be better)

## main

Contains scripts that can be run to train the networks. We use two networks, one
for the grasping ("grasp network") and one for the transition policy ("success
network").

- `train_bed_grasp.py` - trains the grasping network ([hyperparameters here][1])
- `train_bed_success.py` - trains the network that identifies successful
  rollouts ([hyperparameters here][2])

Both of these call `solver = Solver(bed_grasp_options,yolo,pascal)` and then
`solver.train()` to train the network, based on `core/train_network.py`. The
difference is that `yolo` can be the output from `networks/grasp_net_cs.py` or
`networks/success_net.py`. The `pascal` portion handles the YOLO pre-trained
features, so that the two networks can simply fine-tune.

The networks use TF slim. [Documentation here][3].

**Probably better to run for now**: use the scripts `main/grasp.sh` and `main/success.sh` since
these iterate through all indices in a cross validation set, and then put the output to a file, such
as in:

```
./main/grasp.sh | tee logs/grasp.log
```

so that we can inspect the output later.


## src

(This is older stuff from Michael's documentation)

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


[1]:https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/configs/bed_grasp_config.py
[2]:https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/configs/bed_success_config.py
[3]:https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
