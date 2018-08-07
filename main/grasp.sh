
for (( i=0; i<10; i++ )) ; do
    python main/train_bed_grasp.py --seed 1 --do_cv --cv_idx $i \
            --max_iters 2000 --lrate 0.0001 --l2_lambda 0.0001 \
            --dropout_keep_prob 1.0 --fix_pretrained_layers

    python main/train_bed_grasp.py --seed 1 --do_cv --cv_idx $i \
            --max_iters 2000 --lrate 0.0001 --l2_lambda 0.0001 \
            --dropout_keep_prob 0.5 --fix_pretrained_layers

    python main/train_bed_grasp.py --seed 1 --do_cv --cv_idx $i \
            --max_iters 2000 --lrate 0.0001 --l2_lambda 0.0001 \
            --dropout_keep_prob 1.0  --use_smaller_net

    python main/train_bed_grasp.py --seed 1 --do_cv --cv_idx $i \
            --max_iters 2000 --lrate 0.0001 --l2_lambda 0.0001 \
            --dropout_keep_prob 0.5  --use_smaller_net
done

