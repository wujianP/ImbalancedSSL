conda activate /discobox/wjpeng/env/bmb
cd '/discobox/wjpeng/code/ImbalancedSSL/BMB'
DATA='/dev/shm/imagenet'
ANN='/discobox/wjpeng/code/ImbalancedSSL/BMB/dataset/ImageNet_LT'
OUT='/discobox/wjpeng/ckp/BMB/imagenetLT/semi50/ours_adam/test'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29532 eval.py \
 --gpu 4 \
 --dataset imagenet \
 --imb_ratio_l 50 \
 --imb_ratio_u 1 \
 --model resnet_baseline \
 --resume /discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/DASO_50per/DASO_cifar10_l_1500_100_ul_3000_100_seed_34692839/model_best.pth.tar \
 --tcp_strong \
 --tcp_refresh_after_warm \
 --tcp_pool_size 1024 \
 --tcp_get_num 256 \
 --tcp_distribution_type pd_select \
 --tcp_sync_input \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 3 \
 --tcp_sample_power 0.75 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 0.75 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 0.75 \
 --tau 0.7 \
 --epochs 300 \
 --warmup_epochs 10 \
 --num_workers 3 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --labeled_ratio 50 \
 --val_batch_size 128 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 50 \
 --train_log_iter_freq 50 \
 --pd_distribution_estimate_nepoch 999 \
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN}