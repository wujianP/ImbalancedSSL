# FixMatch@r=10
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar100/r10/crest_seed1'
cd /share/home/wjpeng/projects/ImbalancedSSL/CoSSL
python train_cifar_fix_crest.py \
--dataset cifar100 \
--ratio 2 \
--num_max 150 \
--imb_ratio_l 10 \
--imb_ratio_u 10 \
--out ${OUT} \
--manualSeed 1 \
--gpu 1