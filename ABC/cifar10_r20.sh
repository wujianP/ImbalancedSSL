conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/ABC_cifar100_r10_seed2"
cd /discobox/wjpeng/code/ImbalancedSSL/ABC
mkdir $FOLDERDIR

python ABCfix.py \
--gpu 7 \
--manualSeed 2 \
--num_max_l 150 \
--num_max_u 300 \
--dataset cifar100 \
--imb_ratio 10 \
--epoch 500 \
--val-iteration 500 \
--out $FOLDERDIR
