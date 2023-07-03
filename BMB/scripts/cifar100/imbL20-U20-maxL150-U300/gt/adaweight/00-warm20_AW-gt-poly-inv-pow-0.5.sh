cd '/discobox/wjpeng/code/ada-tcp'
DATA='/dev/shm'
OUT='/discobox/wjpeng/ckp/cifar100/imbL20-U20-maxL150-U300/adaweight/warm20_AW-gt-poly-inv-pow-0.5'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 ABCfix.py \
 --ada_weight_type gt \
 --sample_fun_type poly_inv \
 --sample_power 0.5 \
 --tau_high 0.9 \
 --gpu 0,1 \
 --dist_eval \
 --model wideresnet \
 --dataset cifar100 \
 --num_max_l 150 \
 --num_max_u 300 \
 --imb_ratio_l 20 \
 --imb_ratio_u 20 \
 --imb_type long \
 --epoch 250 \
 --warmup_epoch 20 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.004 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 5 \
 --num_workers 6 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 5 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}