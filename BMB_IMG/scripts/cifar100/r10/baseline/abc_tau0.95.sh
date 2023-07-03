cd /discobox/wjpeng/code/Tail-Class-Pool-main/
DATA='/dev/shm'
OUT='/discobox/wjpeng/ckp/cifar100/r10/abc_tau0.95'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29507 ABCfix.py \
 --mask_for_balance_type gt \
 --anneal_mask_for_balance \
 --sample_fun_type poly_inv \
 --sample_power 1 \
 --tau 0.95 \
 --gpu 7 \
 --dist_eval \
 --model wideresnet \
 --dataset cifar100 \
 --num_max_l 150 \
 --num_max_u 300 \
 --imb_ratio_l 10 \
 --imb_ratio_u 10 \
 --imb_type long \
 --epoch 500 \
 --warmup_epoch 20 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 4 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 5 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}