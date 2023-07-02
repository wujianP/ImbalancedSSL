conda activate /discobox/wjpeng/env/bmb/
FOLDERDIR="/discobox/wjpeng/ckp/BMB/rebuttal/baselines/DASO_cifar10_r20_seed1_bs2"
mkdir $FOLDERDIR
cd /discobox/wjpeng/code/ImbalancedSSL/DASO

python main.py \
--config-file configs/cifar10/fixmatch_daso.yaml \
SEED 1 \
GPU_ID 5 \
DATASET.CIFAR10.NUM_LABELED_HEAD 1500 \
DATASET.CIFAR10.NUM_UNLABELED_HEAD 3000 \
DATASET.CIFAR10.IMB_FACTOR_L 20 \
DATASET.CIFAR10.IMB_FACTOR_UL 20 \
SOLVER.UNLABELED_BATCH_RATIO 2 \
OUTPUT_DIR ${FOLDERDIR} \
 | tee $FOLDERDIR/progress.txt
