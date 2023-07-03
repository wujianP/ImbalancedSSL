cd '/share/home/wjpeng/projects/improvedABC'
DATA='/share_io03_ssd/ckpt2/wjpeng/dataset'
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar10/imbL100-U100-maxL1500-U3000/0.5-3-baseline'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 ABCfix.py \
 --resume /share/home/wjpeng/ckp/experiments/TCP/cifar10/imbL100-U100-maxL1500-U3000/0.5-3-baseline/ckps/latest.pth \
 --gpu 0,1 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau_high 0.95 \
 --model wideresnet \
 --dataset cifar10 \
 --num_max_l 1500 \
 --num_max_u 3000 \
 --imb_ratio_l 100 \
 --imb_ratio_u 100 \
 --imb_type long \
 --epoch 250 \
 --warmup_epoch 0 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.004 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 5 \
 --dist_eval \
 --num_workers 4 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 1 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}