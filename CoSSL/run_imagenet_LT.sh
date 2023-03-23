# ImageNet-LT 20% subset
cd /share/home/wjpeng/projects/ImbalancedSSL/CoSSL
DATA='/share/common/ImageDatasets/imagenet_2012'
ANN='/share/home/wjpeng/projects/improvedABC/dataset/ImageNet_LT'
OUT='/share_io03_ssd/ckpt2/wjpeng/experiments/cossl/debug'
#python train_imagenetLT_fix.py \
#--labeled_ratio 20 \
#--epoch 1 \
#--batch_size 64 \
#--val-iteration 10 \
#--out ${OUT}/stage1 \
#--data_path ${DATA} \
#--annotation_file_path ${ANN} \
#--gpu 0

python train_imagenetLT_fix_cossl.py \
--labeled_ratio 20 \
--epoch 40 \
--batch_size 64 \
--val-iteration 10 \
--resume ${OUT}/stage/best.pth \
--out ${OUT}/stage2 \
--data_path ${DATA} \
--annotation_file_path ${ANN} \
--max_lam 0.6 \
--gpu 0
