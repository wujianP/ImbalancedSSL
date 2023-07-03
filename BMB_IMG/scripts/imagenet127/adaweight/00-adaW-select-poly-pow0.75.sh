DATA='/dev/shm/imagenet'
ANN='/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT'
OUT='/discobox/jia/checkpoint/imagenet127_LT/adaw/64-adaW-select-poly-pow0.75'
ROOT=/DDN_ROOT/ytcheng/code/Tail-Class-Pool
cd $ROOT

python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 ABCfix.py \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 0.75 \
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
 --annotation_file_path ${ANN}
