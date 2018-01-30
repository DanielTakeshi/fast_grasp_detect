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
		if i == 0:
			# Add all data from blue white set
			set_id = "bw"
			bw_data_files = os.listdir(sources[i])
			for filename in bw_data_files:
				full_filename = sources[i] + '/' + filename
				num = filename[8:]
				file_dest = dest+'rollout_'+ set_id + '_' +str(num)
				shutil.copytree(full_filename, file_dest)

		if i == 1:
			# Add only "count" points from cal data set
			set_id = "cal"
			for num in arr:
				src = sources[i] + 'rollout_'+str(num)
				file_dest = dest+'rollout_'+ set_id + '_' +str(num)

				while not os.path.exists(src) or os.path.exists(file_dest):
					if (i == 0):
						num = np.random.randint(57)
					else:
						num = np.random.randint(49)
					src = sources[i] + 'rollout_'+str(num)
					file_dest = dest+'rollout_'+ set_id + '_' +str(num)
			
				print("Src: " + str(src))
				print("Dest: " + str(file_dest))
				# if os.path.exists(src) and not os.path.exists(file_dest):
				shutil.copytree(src, file_dest)

	return dest

def make_training_sets(num_cal):
	tl_training_sets = []
	# tl_set_counter = 0
	for i in num_cal:
		blue_white_picks = np.arange(57) # include ALL the blue-white data
		cal_picks = np.random.choice(49, size=i)
		new_training_set = make_rollouts_dir([blue_white_picks, cal_picks], i)
		# tl_set_counter += 1
		tl_training_sets.append(new_training_set)
	return tl_training_sets

num_cal_data = [i*5 for i in range(9)]
ts = make_training_sets(num_cal_data)

