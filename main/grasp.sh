
for (( i=0; i<10; i++ )) ; do
    python main/train_bed_grasp.py --seed 1 --do_cv --cv_idx $i --max_iters 1000 --lrate 0.0001 --l2_lambda 0.0001
done
