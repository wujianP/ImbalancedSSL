# This code is constructed based on Pytorch Implementation of MixMatch(https://github.com/YU1ut/MixMatch-pytorch)

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wrn as models
from dataset import get_imagenet
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize

# from dataset.dataTools import make_imb_data
from dataset.dataTools import make_imb_data, gtDict, prob2Distribution, prepareDataLoaders
from sslAlgo.fixLogger import createLogger, loadLogger, appendLogger, closeLogger
from imbOptim.classWeights import parseClassWeights, createSettings, getClassWeights

from dataset.dojo import dojoTest

parser = argparse.ArgumentParser(description='PyTorch FixMixMatch Training')
# ADD by WJ
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--annotation_file_path', required=True, type=str)
parser.add_argument('--save_freq', type=int, default=30)
parser.add_argument('--labeled_ratio', type=float, default=20, help='by default we take 20% labeled data')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='Weight Decay', help='weight decaying')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.7, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--lambda_u', default=1, type=float, help='Weight-parameter for Unsupervised Loss')

# Hyperparameters for DARP
parser.add_argument('--warm', type=int, default=200,  help='Number of warm up epoch for DARP')
parser.add_argument('--alpha', default=2.0, type=float, help='hyperparameter for removing noisy entries')
parser.add_argument('--darp', action='store_true', help='Applying DARP')
parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')

# Weights for Model's Cost Function
# parser.add_argument('--w_L', choices=["", "default", "total", "minority"], help='Applying Weights to Loss: \
#     \n (default/blank) = Uniform Weight of Ones \
#     \n total = Class Distribution / Total Class Distribution : [1, 3] \
#     \n minority = Class Distribution / Minority Class Distribution : [1, inf]') # Old Arugment
parser.add_argument('--distbu', choices=["", "uniform", "pseudo", \
    "weak", "strong", "gt", "gt_l", "gt_u"], \
    help='Applying Weights to Unsupervised Loss \
    \n (blank/uniform) = Uniform Weight of Ones \
    \n pseudo = Using Pseudo-Label Class Distribution \
    \n weak = Using Weakly Augmented Output Class Distribution \
    \n strong = Using Strongly Augmented Output Class Distribution \
    \n gt = Using Ground Truth Class Distribution (Labeled + Unlabeled) \
    \n gt_l = Using Ground Truth Class Distribution (Labeled) \
    \n gt_u = Using Ground Truth Class Distribution (Unlabeled)') 

parser.add_argument('--distbl', choices=["", "uniform", "gt_l"], \
    help="Applying Weights to Supervised Loss \
        \n (blank/uniform) = Uniform Weight of Ones \
        \n gt_l = Using Ground Truth Class Distribution (Labeled)")

# For Weighting Function Schemes
parser.add_argument('--invert', action='store_true', \
     help='If declared, flip class weights on Loss (Penalize Minority more than Majority)')
parser.add_argument('--normalize', default=None, type=float, \
     help='Normalize class weights on Loss according to number of classes \
         \n such that sum(weights) = num_class * norm_const')
parser.add_argument('--total', default=None, type=float, \
     help='Using Total-Schemed Weights to Unsupervised Loss m*(Class/Total) + 1')
parser.add_argument('--minority', default=None, type=float, \
     help='Using Minority-Schemed Weights to Unsupervised Loss (Class/Minority)')
parser.add_argument('--intercept', default=None, type=float, \
     help='Using Intercept-Schemed Weights to Unsupervised Loss (Class/Total) + b')
parser.add_argument('--log', default=None, type=float, \
     help='Using Minority-Schemed Weights to Unsupervised Loss (log(a*Class)/log(Total))')
parser.add_argument('--effective', default=None, type=float, \
     help='Using Effective Number-Schemed Weights to Unsupervised Loss ((1-beta)/(1-beta^Class)) \
         \n Note: Hyperparameter is automatically calculated')
parser.add_argument('--power', default=None, type=float, \
     help='Using Powered-Schemed Weights to Unsupervised Loss (Total/Class)^alpha')

parser.add_argument('--many_shot_thr', type=int)
parser.add_argument('--low_shot_thr', type=int)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
num_class = 1000 # ImageNet-LT


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing ImageNet-LT')
    tmp = get_imagenet(
        root=args.data_path,
        annotation_file_train_labeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_labeled.txt',
        annotation_file_train_unlabeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_unlabeled.txt',
        annotation_file_val=f'{args.annotation_file_path}/ImageNet_LT_test.txt',
        num_per_class=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_sample_num.txt')
    sample_num_per_class, train_labeled_set, train_unlabeled_set, test_set = tmp

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size= args.batch_size, shuffle=True,
                                            num_workers=6, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=6)
    dataLoaders = {}
    dataLoaders["labeled"] = labeled_trainloader
    dataLoaders["unlabeled"] = unlabeled_trainloader
    dataLoaders["Test"] = test_loader

    N_SAMPLES_PER_CLASS = sample_num_per_class['labeled']
    U_SAMPLES_PER_CLASS = sample_num_per_class['unlabeled']
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)
    
    distb_dict = gtDict(N_SAMPLES_PER_CLASS_T, U_SAMPLES_PER_CLASS, use_cuda) # Collect Ground Truth Distribution

    # Model (Wide ResNet model)
    print("==> creating Resnet50")

    # Used for Fix Match
    def create_model(ema=False):
        from models.resnetwithABC import ResNet
        model = ResNet(num_classes=num_class, encoder_name='resnet', pretrained=False)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # Resume
    title = 'fix-imagenet-lt-20'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['ema_state_dict'])

        # Settings for Weighted loss based on Class Distribution (for Unsupervised)
        class_weight_u, distbLoss_dict, \
            weightLoss = createSettings(num_class, use_cuda, distbu=args.distbu, distbl=args.distbl, invert=args.invert, normalize=args.normalize,
                                        total=args.total, minority=args.minority, intercept=args.intercept, log=args.log,
                                        effective=args.effective, power=args.power)

        class_weight_x = getClassWeights(distbLoss_dict["labeled"], weightLoss, 0, args.darp, distb_dict, use_cuda)
        printSettings(start_epoch, distbLoss_dict, weightLoss)
        loggerDict = createLogger(args.out, num_class, title)

    else:
        raise KeyError

    # Main function

    all_acc, many_acc, medium_acc, low_acc = validate(dataLoaders["Test"], model, criterion, use_cuda, mode='Test Stats ', train_num_per_class=N_SAMPLES_PER_CLASS)

    closeLogger(loggerDict)

    # Print the final results
    print(f'All: {all_acc}, Many: {many_acc}, Medium: {medium_acc}, Low: {low_acc}')


def validate(valloader, model, criterion, use_cuda, mode, train_num_per_class):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class).cuda()
    classwise_num = torch.zeros(num_class).cuda()
    section_acc = torch.zeros(3).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1] # torch.Size([16])
            pred_mask = (targets == pred_label).float() # torch.Size([16])
            
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)

    print(classwise_num[:10])

    ret = shot_accuracy(correct_num_per_class=classwise_correct,
                        many_shot_thr=args.many_shot_thr,
                        low_shot_thr=args.low_shot_thr,
                        train_num_per_class=train_num_per_class,
                        test_num_per_class=classwise_num)

    all_acc, many_acc, medium_acc, low_acc = ret['all_acc'], ret['many_shot_acc'], ret['medium_shot_acc'], ret[
        'low_shot_acc']

    return all_acc, many_acc, medium_acc, low_acc


def shot_accuracy(correct_num_per_class: np.ndarray,
                  train_num_per_class,
                  test_num_per_class,
                  many_shot_thr=100,
                  low_shot_thr=20):
    """
    Args:
        correct_num_per_class: 每个类预测正确的样本数量，如果是dist_eval情况下，需要传入所有GPU上的sum
        train_num_per_class: 训练集中每个类别所含样本数量，用于区分many/medium/low shot
        test_num_per_class: 测试集上每个类别样本数量，与correct_num_per_class配合计算每个类别的准确率，无论dist_eval与否
            都是指的整个测试数据集中每个类别的样本数量
        many_shot_thr:
        low_shot_thr:
    """
    num_class = len(train_num_per_class)
    many_shot_acc = []
    median_shot_acc = []
    low_shot_acc = []
    acc_per_class = []

    for i in range(num_class):
        acc = (correct_num_per_class[i] / test_num_per_class[i]) * 100
        acc_per_class.append(acc.item())
        if train_num_per_class[i] >= many_shot_thr:
            many_shot_acc.append(acc)
        elif train_num_per_class[i] <= low_shot_thr:
            low_shot_acc.append(acc)
        else:
            median_shot_acc.append(acc)

    from IPython import embed
    embed()

    ret = {
        'acc_per_class': np.array(acc_per_class),
        'all_acc': np.mean(acc_per_class),
        'many_shot_acc': torch.tensor(many_shot_acc).mean(),
        'medium_shot_acc': torch.tensor(median_shot_acc).mean(),
        'low_shot_acc': torch.tensor(low_shot_acc).mean()
    }

    return ret

def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d

# To solve KL-Divergence Objective using Newton's Method


def save_checkpoint(state, epoch, save_path, save_freq, is_best=False):

    if epoch % save_freq == 0:
        torch.save(state, os.path.join(save_path, f'checkpoint_{epoch}.pth'))
    # save the best checkpoint
    if is_best:
        if os.path.isfile(os.path.join(save_path, 'best.pth')):
            os.remove(os.path.join(save_path, 'best.pth'))
        torch.save(state, os.path.join(save_path, 'best.pth'))
        print(f'save best ckp')

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask, weights_x, weights_u):
        CE_x = F.log_softmax(outputs_x, dim=1) * targets_x # Cross-Entropy Unsupervised, torch.Size([128, 10])
        WCE_x = CE_x * weights_x     # Weighted Cross-Entropy
        SCE_x = torch.sum(WCE_x, dim=1)     # Summed Cross-Entropy torch.Size([128])
        Lx = -torch.mean(SCE_x)     # Final Unsupervised Cross-Entropy Loss

        CE_u = F.log_softmax(outputs_u, dim=1) * targets_u # Cross-Entropy Unsupervised, torch.Size([128, 10])
        WCE_u = CE_u * weights_u     # Weighted Cross-Entropy
        MCE_u = torch.sum(WCE_u, dim=1) * mask     # Masked Cross-Entropy based on Quality, torch.Size([128])
        Lu = -torch.mean(MCE_u)     # Final Unsupervised Cross-Entropy Loss

        return Lx, Lu


# Weighted Exponential Moving Average
class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def printSettings(start_epoch, distbLoss_dict, weightLoss) :
    print("Starting Epoch: ", start_epoch)
    print("For Weight Loss based on Class Distribution: ")
    print("Class Distribution: ", distbLoss_dict)
    print("Weighting Formula:  ", weightLoss)


if __name__ == '__main__':
    main()
