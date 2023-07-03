cd '/share/home/wjpeng/projects/improvedABC'
DATA='/share_io03_ssd/ckpt2/wjpeng/dataset'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 ABCfix.py \
 --gpu 0,1 \
 --tcp_pool_size 256 \
 --tcp_get_num 128 \
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
 --tau 0.95 \
 --model wideresnet \
 --dataset cifar10 \
 --num_max_l 1500 \
 --num_max_u 3000 \
 --imb_ratio_l 150 \
 --imb_ratio_u 150 \
 --imb_type long \
 --epoch 250 \
 --warmup_epoch 20 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.004 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --dist_eval \
 --num_workers 4 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 1 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}