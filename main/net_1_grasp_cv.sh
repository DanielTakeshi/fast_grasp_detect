
for (( i=0; i<10; i++ )) ; do
    python main/train_bed_grasp.py --do_cv --cv_idx $i \
            --max_iters 8001 --lrate 0.0001 --net_type 1
done

