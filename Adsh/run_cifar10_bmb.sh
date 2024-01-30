conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/test'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh
git pull

python train_fix_cifar10.py \
--gpu-id 2 \
--manualSeed 0 \
--al adsh \
--dataset cifar10 \
--num_classes 10 \
--mu 2 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 1500 \
--label_ratio 2.0 \
--imb_ratio_l 20 \
--imb_ratio_u 20 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt
