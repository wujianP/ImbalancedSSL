conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR='/discobox/wjpeng/ckp/BMB/rebuttal/mismatch/DASO_cifar10_rl50_ru100_seed2'
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/DASO

python main.py \
--config-file configs/cifar10/fixmatch_daso.yaml \
SEED 2 \
GPU_ID 2 \
DATASET.CIFAR10.NUM_LABELED_HEAD 1500 \
DATASET.CIFAR10.NUM_UNLABELED_HEAD 3000 \
DATASET.CIFAR10.IMB_FACTOR_L 50 \
DATASET.CIFAR10.IMB_FACTOR_UL 100 \
SOLVER.UNLABELED_BATCH_RATIO 1 \
OUTPUT_DIR ${FOLDERDIR} \
 | tee $FOLDERDIR/progress.txt
