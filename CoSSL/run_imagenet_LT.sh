# Small-ImageNet@64x64
cd /share/home/wjpeng/projects/ImbalancedSSL/CoSSL
DATA='/share/common/ImageDatasets/imagenet_2012'
ANN='/share/home/wjpeng/projects/improvedABC/dataset/ImageNet_LT'
OUT='/share_io03_ssd/ckpt2/wjpeng/experiments/debug'
python train_imagenetLT_fix.py \
--labeled_ratio 20 \
--epoch 200 \
--batch_size 64 \
--val-iteration 10 \
--out ${OUT} \
--data_path ${DATA} \
--annotation_file_path ${ANN} \
--gpu 0

#python small_imagenet127_fix_cossl.py \
#--labeled_percent 0.1 \
#--img_size 64 \
#--epoch 100 \
#--val-iteration 500 \
#--resume ./results/small_imagenet127_64x64/fixmatch/baseline/resnet50_labeled_percent01/checkpoint_401.pth.tar \
#--out ./results/small_imagenet127_64x64/fixmatch/cossl/resnet50_labeled_percent01 \
#--max_lam 0.6 \
#--gpu 0
