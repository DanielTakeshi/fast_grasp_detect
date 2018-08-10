
for (( i=0; i<1; i++ )) ; do
    python main/train_bed_grasp.py --do_cv --cv_idx $i \
            --max_iters 2000 --lrate 0.0001 --l2_lambda 0.0001 \
            --dropout_keep_prob 1.0 --net_type 3
done

