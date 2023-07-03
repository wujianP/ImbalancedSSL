cd /discobox/wjpeng/code/Tail-Class-Pool-main/
DATA='/dev/shm/imagenet'
ANN='/discobox/wjpeng/code/Tail-Class-Pool-main/dataset/ImageNet_LT'
OUT='/discobox/wjpeng/ckp/imagenetLT/semi20/baseline/abc_warm10'
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 ABCfix.py \
 --mask_for_balance_type gt \
 --anneal_mask_for_balance \
 --sample_fun_type poly_inv \
 --sample_power 1 \
 --gpu 0,1,2,3,4,5,6,7 \
 --dist_eval \
 --disable_ema_model \
 --dataset imagenet \
 --labeled_ratio 20 \
 --model resnet \
 --tau 0.7 \
 --pd_distribution_estimate_nepoch 999 \
 --epochs 200 \
 --warmup_epochs 10 \
 --val-iteration 46 \
 --num_workers 8 \
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
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN}
