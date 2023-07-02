conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/DASO_20per"
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/DASO_imageNet

python main.py \
--config-file configs/fixmatch_daso_imagenetLT.yaml \
GPU_ID 5 \
SOLVER.UNLABELED_BATCH_RATIO 1 \
OUTPUT_DIR ${FOLDERDIR} \
 | tee $FOLDERDIR/progress.txt
