cd '/discobox/wjpeng/code/img127-tcp'
DATA='/dev/shm/imagenet'
ANN='/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT'
OUT='/discobox/wjpeng/ckp/img127/64-adaW/00-pow1.5'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29503 ABCfix.py \
 --gpu 3 \
 --crop_size 64 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1.5 \
 --tau 0.95 \
 --model resnet \
 --dataset imagenet127 \
 --labeled_ratio 10 \
 --epochs 500 \
 --warmup_epoch 0 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 64 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 3 \
 --writer_log_iter_freq 100 \
 --writer_log_class_stride 10 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA} \
 --annotation_file_path ${ANN}