cd '/discobox/wjpeng/code/ada-tcp'
DATA='/dev/shm/imagenet'
ANN='/discobox/wjpeng/code/ada-tcp/dataset/ImageNet_LT'
OUT='/discobox/wjpeng/ckp/imagenetLT/semi50/ours/64-adaW/pow1.5'
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 ABCfix.py \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --gpu 0,1,2,3,4,5,6,7 \
 --dataset imagenet \
 --labeled_ratio 50 \
 --model resnet \
 --dist_eval \
 --find_unused_parameters \
 --disable_ema_model \
 --tau 0.7 \
 --epochs 200 \
 --warmup_epochs 10 \
 --num_workers 8 \
 --val-iteration 114 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 128 \
 --lr 0.2 \
 --lr_scheduler_type cos \
 --optim_type sgd \
 --weight_decay 5e-4 \
 --writer_log_iter_freq 50 \
 --writer_log_class_stride 50 \
 --train_log_iter_freq 50 \
 --pd_distribution_estimate_nepoch 999 \
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN}