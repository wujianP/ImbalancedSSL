conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/Small_ImageNet127/Adsh/res64_1per_8gpu'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh

python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 train_fix_small_imagenet127_ddp.py \
--gpu-ids 0,1,2,3,4,5,6,7 \
--find_unused_parameters \
--lr 0.08 \
--img_size 64 \
--labeled_percent 0.01 \
--al adsh \
--mu 1 \
--total_steps 31250 \
--eval_steps 500 \
--out $FOLDERDIR \
--save_freq 10 \
 | tee ${FOLDERDIR}/progress.txt
