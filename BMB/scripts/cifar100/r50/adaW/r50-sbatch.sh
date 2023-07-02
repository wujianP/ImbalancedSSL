#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks 16
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --gres=gpu:3090:4
#SBATCH --mem=160G
#SBATCH --job-name cifar100-r50
#SBATCH -o /share_io03_ssd/ckpt2/wjpeng/experiments/TCP/cifar100/r50/%j.out
#SBATCH -e /share_io03_ssd/ckpt2/wjpeng/experiments/TCP/cifar100/r50/%j.err

pwd
nvidia-smi
cd '/share/home/wjpeng/projects/improvedABC/scripts/cifar100/r50/64-adaW'
sh pow0.75.sh &
sh pow1.sh &
sh pow1.5.sh &
sh pow1.75.sh &
