conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/Adsh_cifar100_r10_seed1'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python train_fix_cifar10.py \
--gpu-id 7 \
--manualSeed 1 \
--al adsh \
--dataset cifar100 \
--num_classes 100 \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 150 \
--label_ratio 2.0 \
--imb_ratio_l 10 \
--imb_ratio_u 10 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt
