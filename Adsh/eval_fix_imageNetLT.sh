conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/Adsh_50per_test'
DATA_PATH='/dev/shm/imagenet'
ANN_PATH='/discobox/wjpeng/code/img127-tcp/dataset/ImageNet_LT'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python eval_fix_imageNetLT.py \
--labeled_ratio 20 \
--many_shot_thr 50 \
--low_shot_thr 10 \
--resume /discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/Adsh_50per/best.pth  \
--gpu-id 5 \
--gpu 5 \
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
