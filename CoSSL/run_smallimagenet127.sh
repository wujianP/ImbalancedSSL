conda activate /discobox/wjpeng/env/py38
cd '/discobox/wjpeng/code/ImbalancedSSL/CoSSL'
OUT='/discobox/wjpeng/ckp/img127-32/baseline/cossl_1_percent'
python bmb_train_small_imagenet127_fix.py \
--labeled_percent 0.01 \
--img_size 32 \
--epoch 401 \
--val-iteration 500 \
--batch_size 64 \
--lr 0.002 \
--out ${OUT}/stage1 \
--gpu 0

python bmb_train_small_imagenet127_fix_cossl.py \
--labeled_percent 0.01 \
--img_size 32 \
--epoch 100 \
--val-iteration 500 \
--batch_size 64 \
--lr 0.002 \
--lr_tfe 0.002 \
--resume ${OUT}/stage1/checkpoint_400.pth \
--out ${OUT}/stage2 \
--max_lam 0.6 \
--gpu 0
