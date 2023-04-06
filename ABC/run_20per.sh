cd '/share/home/wjpeng/projects/ImbalancedSSL/ABC'
DATA='/share/common/ImageDatasets/imagenet_2012'
ANN='/share/home/wjpeng/projects/improvedABC/dataset/ImageNet_LT'
OUT='/share_io03_ssd/ckpt2/wjpeng/experiments/debug'
python train_imagenetLT_fix_abc.py \
--gpu 0 \
--labeled_ratio 20 \
--epoch 300 \
--lr 0.002 \
--batch_size 64 \
--val-iteration 20 \
--out ${OUT} \
--data_path ${DATA} \
--annotation_file_path ${ANN}