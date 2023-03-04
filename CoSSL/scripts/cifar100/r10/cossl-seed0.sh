# FixMatch@r=10
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar100/r10/cossl/seed0'
cd ../../..
python train_cifar_fix.py \
--dataset cifar100 \
--ratio 2 \
--num_max 150 \
--imb_ratio_l 10 \
--imb_ratio_u 10 \
--epoch 500 \
--val-iteration 500 \
--out ${OUT}/stage1 \
--manualSeed 0 \
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
--manualSeed 0 \
--gpu 0