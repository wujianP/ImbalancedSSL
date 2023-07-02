conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/DASO_cifar100_r10_seed1_bs1"
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/DASO

python main.py \
--config-file configs/cifar100/fixmatch_daso.yaml \
SEED 1 \
GPU_ID 0 \
DATASET.CIFAR10.NUM_LABELED_HEAD 150 \
DATASET.CIFAR10.NUM_UNLABELED_HEAD 300 \
DATASET.CIFAR10.IMB_FACTOR_L 10 \
DATASET.CIFAR10.IMB_FACTOR_UL 10 \
SOLVER.UNLABELED_BATCH_RATIO 1 \
OUTPUT_DIR ${FOLDERDIR} \
 | tee $FOLDERDIR/progress.txt
