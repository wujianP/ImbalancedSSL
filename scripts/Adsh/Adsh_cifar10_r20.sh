FOLDERDIR='/share/home/wjpeng/ckp/BMB/rebuttal/baselines/Adsh_cifar10_r20_seed0'
mkdir $FOLDERDIR
cd /share/home/wjpeng/projects/ImbalancedSSL/Adsh

python train_fix_cifar10.py \
--gpu-id 1 \
--manualSeed 0 \
--al adsh \
--dataset cifar10 \
--num_classes 10 \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 1500 \
--label_ratio 2.0 \
--imb_ratio_l 20 \
--imb_ratio_u 20 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt

FOLDERDIR='/share/home/wjpeng/ckp/BMB/rebuttal/baselines/Adsh_cifar10_r20_seed1'
mkdir $FOLDERDIR
cd /share/home/wjpeng/projects/ImbalancedSSL/Adsh

python train_fix_cifar10.py \
--gpu-id 1 \
--manualSeed 1 \
--al adsh \
--dataset cifar10 \
--num_classes 10 \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 1500 \
--label_ratio 2.0 \
--imb_ratio_l 20 \
--imb_ratio_u 20 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt

 FOLDERDIR='/share/home/wjpeng/ckp/BMB/rebuttal/baselines/Adsh_cifar10_r20_seed2'
mkdir $FOLDERDIR
cd /share/home/wjpeng/projects/ImbalancedSSL/Adsh

python train_fix_cifar10.py \
--gpu-id 1 \
--manualSeed 2 \
--al adsh \
--dataset cifar10 \
--num_classes 10 \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--num_max 1500 \
--label_ratio 2.0 \
--imb_ratio_l 20 \
--imb_ratio_u 20 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt
