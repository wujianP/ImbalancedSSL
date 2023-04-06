conda activate /discobox/wjpeng/env/py38
cd '/discobox/wjpeng/code/img127-tcp'
DATA='/dev/shm/small_imagenet127'
OUT='/discobox/wjpeng/ckp/img127-32/adaW1_s256g128bp1sp1wt0.75_1node'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29506 ABCfix.py \
 --gpu 6 \
 --crop_size 32 \
 --tcp_strong \
 --tcp_sync_input \
 --tcp_refresh_after_warm \
 --tcp_pool_size 256 \
 --tcp_get_num 128 \
 --tcp_distribution_type pd_select \
 --tcp_sample_fun_type poly_inv \
 --tcp_balance_power 1 \
 --tcp_sample_power 1 \
 --tcp_put_type inpool \
 --tcp_remove_type inpool \
 --tcp_loss_weight 0.75 \
 --ada_weight_type pd_select \
 --sample_fun_type poly_inv \
 --sample_power 1 \
 --tau 0.95 \
 --model resnet_img127 \
 --dataset small_imagenet127 \
 --labeled_ratio 10 \
 --epochs 500 \
 --warmup_epoch 20 \
 --val-iteration 500 \
 --labeled_batch_size 64 \
 --unlabeled_batch_size 64 \
 --val_batch_size 128 \
 --lr 0.002 \
 --lr_scheduler_type none \
 --optim_type adam \
 --pd_distribution_estimate_nepoch 999 \
 --num_workers 4 \
 --writer_log_iter_freq 1000 \
 --writer_log_class_stride 1 \
 --train_log_iter_freq 100 \
 --find_unused_parameters \
 --out ${OUT} \
 --data_path ${DATA}


conda activate /discobox/wjpeng/env/py38
cd '/discobox/wjpeng/code/img127-tcp'
DATA='/dev/shm/small_imagenet127'
OUT='/discobox/wjpeng/ckp/img127-32/adaW1_s256g128bp1sp1wt0.75_1node'
python small_imagenet127_fix.py --labeled_percent 0.1 --img_size 32 --epoch 500 --val-iteration 500 --out ./results/small_imagenet127_32x32/fixmatch/baseline/resnet50_labeled_percent01 --gpu 0
python small_imagenet127_fix_cossl.py --labeled_percent 0.1 --img_size 32 --epoch 100 --val-iteration 500 --resume ./results/small_imagenet127_32x32/fixmatch/baseline/resnet50_labeled_percent01/checkpoint_401.pth.tar --out ./results/small_imagenet127_32x32/fixmatch/cossl/resnet50_labeled_percent01 --max_lam 0.6 --gpu 0