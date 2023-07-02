# FixMatch@r=10
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar100/r10/cossl/seed1'
cd /share/home/wjpeng/projects/ImbalancedSSL/CoSSL
python train_cifar_fix.py \
--dataset cifar100 \
--ratio 2 \
--num_max 150 \
--imb_ratio_l 10 \
--imb_ratio_u 10 \
--epoch 402 \
--val-iteration 500 \
--out ${OUT}/stage1 \
--manualSeed 1 \
--gpu 0

python train_cifar_fix_cossl.py \
--dataset cifar100 \
--ratio 2 \
--num_max 150 \
--imb_ratio_l 10 \
--imb_ratio_u 10 \
--epoch 100 \
--val-iteration 500 \
--resume ${OUT}/stage1/checkpoint_401.pth.tar \
--out ${OUT}/stage2 \
--max_lam 0.6 \
--manualSeed 1 \
--gpu 0