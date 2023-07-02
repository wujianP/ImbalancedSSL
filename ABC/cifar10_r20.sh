conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/ABC_cifar10_r20_seed0"
cd /discobox/wjpeng/code/ImbalancedSSL/ABC
mkdir $FOLDERDIR

python ABCfix.py \
--gpu 2 \
--manualSeed 0 \
--num_max_l 1500 \
--num_max_u 3000 \
--dataset cifar10 \
--imb_ratio 20 \
--epoch 500 \
--val-iteration 500 \
--out $FOLDERDIR
