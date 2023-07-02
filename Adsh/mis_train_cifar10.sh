conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/mismatch/Adsh_cifar10_rl50_ru100_seed2'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python train_fix_cifar10.py \
--gpu-id 1 \
--gpu 1 \
--manualSeed 2 \
--al adsh \
--dataset cifar10 \
--num_classes 10 \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 1500 \
--label_ratio 2.0 \
--imb_ratio_l 50 \
--imb_ratio_u 100 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt
