cd /discobox/wjpeng/code/ada-tcp
DATA='/dev/shm/imagenet'
ANN='/discobox/wjpeng/code/ada-tcp/dataset/ImageNet_LT'
OUT='/discobox/wjpeng/ckp/imagenetLT/semi20/baseline/supervised_only'
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 ABCfix.py \
 --gpu 0,1,2,3,4,5,6,7 \
 --dataset imagenet \
 --labeled_ratio 20 \
 --model resnet \
 --dist_eval \
 --find_unused_parameters \
 --eval_base \
 --disable_ema_model \
 --disable_abc \
 --loss_u_weight 0 \
 --tcp_loss_weight 0 \
 --epochs 200 \
 --num_workers 8 \
 --val-iteration 46 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 8 \
 --val_batch_size 128 \
 --lr 0.2 \
 --lr_scheduler_type cos \
 --optim_type sgd \
 --weight_decay 5e-4 \
 --writer_log_iter_freq 50 \
 --writer_log_class_stride 50 \
 --train_log_iter_freq 50 \
 --pd_distribution_estimate_nepoch 5 \
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN}