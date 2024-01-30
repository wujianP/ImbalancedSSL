conda activate /discobox/wjpeng/env/bmb/
cd /discobox/wjpeng/code/ImbalancedSSL/Adsh
git pull

NAME=Adsh_cifar10_r20_tcp-size128-get64-bp2-sp1-wt0.5
FOLDERDIR=/discobox/wjpeng/ckp/BMB/rebuttal/combine/$NAME
mkdir $FOLDERDIR

export CUDA_VISIBLE_DEVICES=7
python train_fix_cifar10.py \
--gpu-id 7 \
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
--pool_size 128 \
--get_num 64 \
--wandb_name $NAME \
--wandb_project_name Adsh_BMB_CIFAR10 \
--bp_power 2 \
--sp_power 1 \
--bmb_loss_wt 1

