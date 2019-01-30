# Scripts

## Typical Usage

Run these like: `./main/script_name.sh`.

I use 6000 iterations for the `combo` data, and 4000 iterations for individual data.


## Ablation Study

For the new ablation study where we look at performance based on different sized
training data, run `./main/ablation_training.size.sh` script. But there is a
little more manual work we have to do.

First, we will assign the 0th `cache_combo_v03` as the TEST set. Then try
different combinations of data: 1 through 9 (inclusive), 2 through 9
(inclusive), and so on, up to just 9 through 9 (inclusive), giving us 9
different training data sizes

Unfortunately this requires some manual work. First, we need to fix the data
manger at `core/data_manager.py` where it has:

```python
# Load data, following logic in our configuration file. Test set = held out.
# With CV, make a length-1 list to maintain compatibility with non-CV case.
num = len(cfg.CV_GROUPS)
if cfg.PERFORM_CV:
    cidx = cfg.CV_HELD_OUT_INDEX
    self.training_list = sorted(
            [join(cfg.ROLLOUT_PATH, cfg.CV_GROUPS[c]) for c in range(num) if c != cidx]
    )
    self.held_out_list = [ join(cfg.ROLLOUT_PATH, cfg.CV_GROUPS[cidx]) ]
else:
```

I think all we need to do is set `self.training_list` to:

```
self.training_list = self.training_list[0:]   # for training data 1 through 9
self.training_list = self.training_list[1:]   # for training data 2 through 9
...
self.training_list = self.training_list[8:]   # for training data 9 through 9
```

(Remember, `self.training_list` doesn't include the 0 if we make it the cv index.)

We would need to do this manually before each run.

OK, then run the ablation script.

AFTER this, we re-name the files located at:

```
/nfs/diskstation/seita/bed-make/grasp/cache_combo_v03
```

where we save the files. Rename the directories so we know which training data
we used, i.e., add to the end `1_to_9`, `1_to_8`, etc.

Then run the script `ablation.py` in `IL_ROS_HSR` which will handle the case
for plotting different data points.

I think things will be fine if we use the same number of 'iterations' that we
have, because that assumes iterations x batch-size total elements encountered
for gradient updates. That's fairer than using epochs since different training
data sizes will result in their epochs consisting of different total elements.
