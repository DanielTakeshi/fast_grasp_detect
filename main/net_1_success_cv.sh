
for (( i=0; i<10; i++ )) ; do
    python main/train_bed_success.py --do_cv --cv_idx $i \
            --max_iters 3000 --lrate 0.0001 --net_type 1
done

