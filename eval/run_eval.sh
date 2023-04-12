cd '/discobox/wjpeng/code/img127-tcp'
DATA='/dev/shm/small_imagenet127'
OUT='/discobox/wjpeng/ckp/eval'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29599 eval.py \
 --crop_size 32 \
 --gpu 7 \
 --tau 0.95 \
 --labeled_ratio 10 \
 --model resnet_img127 \
 --dataset small_imagenet127 \
 --epoch 500 \
 --warmup_epoch 10 \
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
 --writer_log_class_stride 1 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA} \
 --resume /discobox/wjpeng/ckp/img127-32/baseline/abc_10_percent/best.pth
