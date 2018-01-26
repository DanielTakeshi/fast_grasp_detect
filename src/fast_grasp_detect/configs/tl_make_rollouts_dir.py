import os
import shutil
import numpy as np

ROOT_DIR = '/media/autolab/1tb/data/'
DATA_PATH = ROOT_DIR + 'bed_rcnn/'

sources = [DATA_PATH+'rollouts_dart/', DATA_PATH+'rollouts_dart_cal/']
tl_experiments_dir = DATA_PATH + 'rollouts_tl/'
if not os.path.exists(tl_experiments_dir):
	os.makedirs(tl_experiments_dir)
TL_PATH = tl_experiments_dir


def make_rollouts_dir(file_nums, count):
	assert len(file_nums) == len(sources)

	dest = TL_PATH + 'rollouts_tl_' + str(count) + '/'
	if not os.path.exists(dest):
		os.makedirs(dest)

	for i, arr in enumerate(file_nums):
		for num in arr:
			src = sources[i] + 'rollout_'+str(num)
			print(src)
			print(dest)
			shutil.copytree(src, dest+'rollout_'+str(num))

	return dest

def make_training_sets(n):
	tl_training_sets = []
	tl_set_counter = 0
	for i in range(n):
		blue_white_picks = np.random.randint(57, size=28)
		cal_picks = np.random.randint(49, size=24)
		new_training_set = make_rollouts_dir([blue_white_picks, cal_picks], tl_set_counter)
		tl_set_counter += 1
		tl_training_sets.append(new_training_set)
	return tl_training_sets

ts = make_training_sets(5)

