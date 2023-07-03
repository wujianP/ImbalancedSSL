# FixMatch-cifar10@rl=100,ru=50
OUT='/share/home/wjpeng/ckp/experiments/TCP/cifar10/rl100-ru50/cossl_seed0'
cd /share/home/wjpeng/projects/ImbalancedSSL/CoSSL
#python train_cifar_fix.py \
# --dataset cifar10 \
# --ratio 2 \
# --num_max 1500 \
# --imb_ratio_l 100 \
# --imb_ratio_u 50 \
# --epoch 402 \
# --val-iteration 500 \
# --out ${OUT}/stage1 \
# --manualSeed 0 \
# --gpu 1
python train_cifar_fix_cossl.py \
--dataset cifar10 \
--ratio 2 \
--num_max 1500 \
--imb_ratio_l 100 \
--imb_ratio_u 50 \
--epoch 100 \
--val-iteration 500 \
--resume ${OUT}/stage1/checkpoint_401.pth.tar \
--out ${OUT}/stage2 \
--max_lam 0.6 \
--manualSeed 0 \
--gpu 1
