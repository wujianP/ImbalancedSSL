cd '/discobox/wjpeng/code/ada-tcp'
DATA='/dev/shm'
OUT='/discobox/wjpeng/ckp/cifar100/imbL20-U20-maxL150-U300/adaweight+tcp/best/Tcp-s1024-g512-select-sync-poly-bpow2-spow1'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 ABCfix.py \
 --gpu 4,5 \
 --tcp_pool_size 1024 \
 --tcp_get_num 512 \
 --tcp_distribution_type pd_select \
 --tcp_sync_input \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 2 \
 --tcp_sample_power 1 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau 0.9 \
 --dist_eval \
 --model wideresnet \
 --dataset cifar100 \
 --num_max_l 150 \
 --num_max_u 300 \
 --imb_ratio_l 20 \
 --imb_ratio_u 20 \
 --imb_type long \
 --epoch 250 \
 --warmup_epoch 0 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.004 \
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