# Resolution 32 x 32, 1% labeled set
conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/baselines/Small_ImageNet127/SAW/res64_1per_8node'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/SAW
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 train_fix_small_imagenet127_ddp.py \
--gpu_ids 0,1,2,3,4,5,6,7 \
--find_unused_parameters \
--img_size 64 \
--labeled_percent 0.01 \
--epochs 250 \
--start-epoch 0 \
--batch-size 64 \
--lr 0.008 \
--val-iteration 125 \
--tau 0.95 \
--ema-decay 0.999 \
--lambda_u 1 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--effective 0.99 \
--distbl gt_l \
--distbu pseudo \
--normalize 1 \
--out $FOLDERDIR \
 | tee ${FOLDERDIR}/progress.txt

