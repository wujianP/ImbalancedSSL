cd '/share/home/wjpeng/projects/improvedABC'
DATA='/share_io03_ssd/ckpt2/wjpeng/dataset'
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar100/r20/Tcp-s256g128_bp2_sp1_wt0.25-warm5E'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29500 ABCfix.py \
 --tcp_pool_size 256 \
 --tcp_get_num 128 \
 --tcp_distribution_type pd_select \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 2 \
 --tcp_sample_power 1 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 0.25 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.75 \
 --tau 0.9 \
 --gpu 0 \
 --dist_eval \
 --model wideresnet \
 --dataset cifar100 \
 --num_max_l 150 \
 --num_max_u 300 \
 --imb_ratio_l 20 \
 --imb_ratio_u 20 \
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
 --pd_distribution_estimate_nepoch 5 \
 --num_workers 4 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 5 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}