conda activate /discobox/wjpeng/env/bmb/
CUDA_LAUNCH_BLOCKING=1
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/imagenetLT/DASO_20per"
DATA_PATH='/dev/shm/imagenet'
ANN_PATH='/discobox/wjpeng/code/img127-tcp/dataset/ImageNet_LT'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/DASO_imageNet

python main.py \
--config-file configs/fixmatch_daso_imagenetLT.yaml \
DATASET.LABELED_RATIO 20 \
DATASET.NAME imagenetLT \
GPU_ID 4 \
SOLVER.MAX_ITER 15000 \
PERIODS.EVAL 500 \
ALGORITHM.CONFIDENCE_THRESHOLD 0.7 \
SOLVER.UNLABELED_BATCH_RATIO 1 \
OUTPUT_DIR ${FOLDERDIR} \
MODEL.QUEUE.FEAT_DIM 2048 \
MODEL.NUM_CLASSES 1000 \
 | tee $FOLDERDIR/progress.txt
