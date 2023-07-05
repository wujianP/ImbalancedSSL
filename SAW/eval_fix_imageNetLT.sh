conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/SAW_50per_test'
DATA_PATH='/dev/shm/imagenet'
ANN_PATH='/discobox/wjpeng/code/img127-tcp/dataset/ImageNet_LT'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/SAW
python eval_fix_imageNetLT.py \
--gpu 5 \
--many_shot_thr 50 \
--low_shot_thr 10 \
--resume /discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/SAW_50per/best.pth  \
--labeled_ratio 20 \
--epochs 300 \
--start-epoch 0 \
--batch-size 64 \
--lr 0.002 \
--val-iteration 500 \
--tau 0.7 \
--ema-decay 0.999 \
--lambda_u 1 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--out $FOLDERDIR \
--data_path $DATA_PATH \
--annotation_file_path $ANN_PATH \
 | tee ${FOLDERDIR}/progress.txt