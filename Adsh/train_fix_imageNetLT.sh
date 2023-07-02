conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/Adsh_20per'
DATA_PATH='/dev/shm/imagenet'
ANN_PATH='/discobox/wjpeng/code/img127-tcp/dataset/ImageNet_LT'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python train_fix_imageNetLT.py \
--labeled_ratio 20 \
--gpu-id 2 \
--gpu 2 \
--manualSeed 0 \
--al adsh \
--mu 1 \
--total_steps 150000 \
--eval_steps 500 \
--out $FOLDERDIR \
--data_path $DATA_PATH \
--annotation_file_path $ANN_PATH \
--save_freq 10 \
 | tee ${FOLDERDIR}/progress.txt
