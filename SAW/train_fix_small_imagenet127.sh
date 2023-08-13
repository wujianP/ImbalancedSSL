# Resolution 32 x 32, 1% labeled set
conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/Small_ImageNet127/SAW_res64_10per'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/SAW
python train_fix_small_imagenet127.py \
--gpu 2 \
--img_size 64 \
--labeled_percent 0.01 \
--epochs 500 \
--start-epoch 0 \
--batch-size 64 \
--lr 0.002 \
--val-iteration 500 \
--tau 0.95 \
--ema-decay 0.999 \
--lambda_u 1 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt

