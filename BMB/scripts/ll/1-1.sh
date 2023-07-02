cd '/share/home/wjpeng/projects/ablateBMB'
DATA='/share_io03_ssd/ckpt2/wjpeng/dataset'
OUT='/share/home/wjpeng/ckp/experiments/bmb/ablate/memory-size-cifar100/s256g64_seed0'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 ABCfix.py \
 --gpu 1 \
 --tcp_pool_size 256 \
 --tcp_get_num 64 \
 --tcp_sample_power 1.25 \
 --tcp_strong \
 --tcp_refresh_after_warm \
 --tcp_distribution_type pd_select \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 3 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 1.25 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau 0.95 \
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
 --val_batch_size 128 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 3 \
 --writer_log_iter_freq 500\
 --writer_log_class_stride 1\
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}

 cd '/share/home/wjpeng/projects/ablateBMB'
DATA='/share_io03_ssd/ckpt2/wjpeng/dataset'
OUT='/share/home/wjpeng/ckp/experiments/bmb/ablate/memory-size-cifar100/s256g256_seed0'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 ABCfix.py \
 --gpu 1 \
 --tcp_pool_size 256 \
 --tcp_get_num 256 \
 --tcp_sample_power 1.25 \
 --tcp_strong \
 --tcp_refresh_after_warm \
 --tcp_distribution_type pd_select \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 3 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 1.25 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau 0.95 \
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
 --val_batch_size 128 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 3 \
 --writer_log_iter_freq 500\
 --writer_log_class_stride 1\
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}
