cd '/discobox/wjpeng/code/ImbalancedSSL/CoSSL'
DATA='/dev/shm/imagenet'
ANN='/discobox/wjpeng/code/img127-tcp/dataset/ImageNet_LT'
OUT='/discobox/wjpeng/ckp/BMB/imagenetLT/semi20/baseline/cossl_lr0.002_500ep_iter148_bs64'
python bmb_train_imagenetLT_fix.py \
--labeled_ratio 20 \
--epoch 401 \
--lr 0.002 \
--batch_size 64 \
--val-iteration 148 \
--out ${OUT}/stage1 \
--data_path ${DATA} \
--annotation_file_path ${ANN} \
--gpu 0

python bmb_train_imagenetLT_fix_cossl.py \
--labeled_ratio 20 \
--lr 0.002 \
--lr_tfe 0.002 \
--epoch 100 \
--batch_size 64 \
--val-iteration 148 \
--resume ${OUT}/stage1/checkpoint_400.pth \
--out ${OUT}/stage2 \
--data_path ${DATA} \
--annotation_file_path ${ANN} \
--max_lam 0.6 \
--gpu 0
