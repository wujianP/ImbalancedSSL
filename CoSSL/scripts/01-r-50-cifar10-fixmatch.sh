# FixMatch@r=50
cd ../
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar10/fixmatch/cossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0