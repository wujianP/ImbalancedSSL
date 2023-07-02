conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/mismatch/ABC_cifar10_rl50_ru1_seed1'
cd /discobox/wjpeng/code/ImbalancedSSL/ABC
mkdir $FOLDERDIR

python ABCfix.py \
--gpu 4 \
--manualSeed 1 \
--num_max_l 1500 \
--num_max_u 3000 \
--dataset cifar10 \
--imb_ratio_l 50 \
--imb_ratio_u 1 \
--epoch 500 \
--val-iteration 500 \
--out $FOLDERDIR
