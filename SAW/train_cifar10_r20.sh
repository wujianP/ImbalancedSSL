conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/SAW_cifar10_r20_seed2'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/SAW
python train_fix_cifar10.py \
--manualSeed 2 \
--gpu 2 \
--epochs 500 \
--start-epoch 0 \
--batch-size 64 \
--lr 0.002 \
--wd 0 \
--num_max 1500 \
--ratio 2.0 \
--imb_ratio_l 20 \
--imb_ratio_u 20 \
--val-iteration 500 \
--num_val 10 \
--tau 0.95 \
--ema-decay 0.999 \
--lambda_u 1 \
--warm 200 \
--alpha 2.0 \
--iter_T 10 \
--num_iter 10 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt
