DATA='/dev/shm/imagenet'
ANN='/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT'
OUT='/discobox/jia/checkpoint/imagenet127_LT/adaw+tcp/64-adaW-select-poly-pow1.0_uweight2.0_tcp2048_g512_bp3_sp0.25_wt0.5'
ROOT=/DDN_ROOT/ytcheng/code/Tail-Class-Pool
cd $ROOT

python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 ABCfix.py \
 --tcp_pool_size 2048 \
 --tcp_get_num 512 \
 --tcp_distribution_type pd_select \
 --tcp_sync_input \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 3 \
 --tcp_sample_power 0.25 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 0.5 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.0 \
 --gpu 0,1,2,3,4,5,6,7 \
 --dataset imagenet127 \
 --labeled_ratio 10 \
 --model resnet \
 --dist_eval \
 --find_unused_parameters \
 --disable_ema_model \
 --tau 0.7 \
 --epochs 200 \
 --warmup_epochs 10 \
 --num_workers 8 \
 --val-iteration 250 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 128 \
 --lr 0.2 \
 --lr_scheduler_type cos \
 --optim_type sgd \
 --weight_decay 5e-4 \
 --writer_log_iter_freq 50 \
 --writer_log_class_stride 127 \
 --train_log_iter_freq 50 \
 --pd_distribution_estimate_nepoch 999 \
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN} \
 --loss_u_weight 2.0 \
