cd '/discobox/wjpeng/code/ada-tcp'
DATA='/dev/shm'
OUT='/discobox/wjpeng/ckp/cifar100/r20/05-pow2-warm20-tau0.95.sh'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29505 ABCfix.py \
 --gpu 5 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 2 \
 --tau 0.95 \
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
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 4 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 5 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}