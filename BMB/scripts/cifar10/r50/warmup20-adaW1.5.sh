cd '/discobox/wjpeng/code/ada-tcp'
DATA='/dev/shm'
OUT='/discobox/wjpeng/ckp/cifar10/r50/warmup20-adaW1.5'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 ABCfix.py \
 --gpu 2 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau 0.95 \
 --model wideresnet \
 --dataset cifar10 \
 --num_max_l 1500 \
 --num_max_u 3000 \
 --imb_ratio_l 50 \
 --imb_ratio_u 50 \
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
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 8 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 1 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}