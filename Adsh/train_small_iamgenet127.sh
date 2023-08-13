conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/Small_ImageNet127/Adsh_res64_10per'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python train_fix_small_imagenet127.py \
--gpu-id 7 \
--gpu 7 \
--img_size 64 \
--labeled_percent 0.1 \
--manualSeed 0 \
--al adsh \
--mu 1 \
--total_steps 250000 \
--eval_steps 500 \
--out $FOLDERDIR \
--save_freq 10 \
 | tee ${FOLDERDIR}/progress.txt
