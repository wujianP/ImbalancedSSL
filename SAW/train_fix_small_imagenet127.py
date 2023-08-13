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
import dataset.fix_cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize

# from dataset.dataTools import make_imb_data
from dataset.dataTools import make_imb_data, gtDict, prob2Distribution, prepareDataLoaders
from sslAlgo.fixLogger import createLogger, loadLogger, appendLogger, closeLogger
from imbOptim.classWeights import parseClassWeights, createSettings, getClassWeights

from dataset.dojo import dojoTest

parser = argparse.ArgumentParser(description='PyTorch FixMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='Weight Decay', help='weight decaying')
parser.add_argument('--save_freq', type=int, default=20)
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
parser.add_argument('--labeled_percent', type=float, default=0.1, help='by default we take 10% labeled data')
parser.add_argument('--img_size', type=int, default=32, help='ImageNet127_32 or ImageNet127_64')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')
parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation data')

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
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
num_class = 127 # CIFAR-10


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing small ImageNet127')
    img_size2path = {32: '/dev/shm/small_imagenet127/res32', 64: '/dev/shm/small_imagenet127/res64'}
    from dataset import get_small_imagenet
    tmp = get_small_imagenet(img_size2path[args.img_size], args.img_size, labeled_percent=args.labeled_percent,
                             seed=args.manualSeed, return_strong_labeled_set=False)

    SAMPLES_PER_CLASS, train_labeled_set, train_unlabeled_set, test_set = tmp

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size= args.batch_size, shuffle=True,
                                            num_workers=8, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    dataLoaders = {}
    dataLoaders["labeled"] = labeled_trainloader
    dataLoaders["unlabeled"] = unlabeled_trainloader
    dataLoaders["Test"] = test_loader

    N_SAMPLES_PER_CLASS = SAMPLES_PER_CLASS['labeled']
    U_SAMPLES_PER_CLASS = SAMPLES_PER_CLASS['unlabeled']
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)
    
    distb_dict = gtDict(N_SAMPLES_PER_CLASS_T, U_SAMPLES_PER_CLASS, use_cuda) # Collect Ground Truth Distribution

    # Model (Wide ResNet model)
    print("==> creating ResNet50")

    # Used for Fix Match
    def create_model(ema=False):
        import models.resnet as models
        model = models.ResNet50(num_classes=num_class, rotation=True, classifier_bias=True)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay) # Exponential Moving Avg
    start_epoch = 0

    # Resume
    title = 'fix-small-imagenet127'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out, start_epoch, model, ema_model,\
            optimizer, class_weight_x, class_weight_u, distbLoss_dict,\
                weightLoss = loadCheckpoint(args.resume, model, ema_model, optimizer)
        print("Resuming on Folder: ", args.out)

        printSettings(start_epoch, distbLoss_dict, weightLoss)
        loggerDict = loadLogger(args.out, title)

    else:
        # Settings for Weighted loss based on Class Distribution (for Unsupervised)
        class_weight_u, distbLoss_dict,\
            weightLoss = createSettings(num_class, use_cuda, \
                                        distbu=args.distbu, distbl=args.distbl, \
                                        invert=args.invert, normalize=args.normalize, \
                                        total=args.total, minority=args.minority,  \
                                            intercept=args.intercept, log=args.log, \
                                                effective=args.effective, power=args.power)
        
        class_weight_x = getClassWeights(distbLoss_dict["labeled"], weightLoss, 0, args.darp, distb_dict, use_cuda)
        printSettings(start_epoch, distbLoss_dict, weightLoss)
        loggerDict = createLogger(args.out, num_class, title)

    test_accs = []
    test_gms = []

    # Default values for ReMixMatch and DARP
    emp_distb_u = torch.ones(num_class) / num_class
    pseudo_orig = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class
    pseudo_refine = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class

    # Lambda_u scheduler
    prev_train_loss = 10000
    lambda_u = args.lambda_u

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Use the estimated distribution of unlabeled data
        if args.est:
            est_name = './estimation/cifar10@N_1500_r_{}_{}_estim.npy'.format(args.imb_ratio_l, args.imb_ratio_u)
            est_disb = np.load(est_name)
            target_disb = sum(U_SAMPLES_PER_CLASS) * torch.Tensor(est_disb) / np.sum(est_disb)
        # Use the inferred distribution with labeled data
        else:
            target_disb = N_SAMPLES_PER_CLASS_T * sum(U_SAMPLES_PER_CLASS) / sum(N_SAMPLES_PER_CLASS)

        # In case of FixMatch, labeled data is utilized as unlabeled data once again.
        target_disb += N_SAMPLES_PER_CLASS_T

        train_loss, train_loss_x, train_loss_u, \
            emp_distb_u, pseudo_orig, pseudo_refine, \
                distb_dict["pseudo"], distb_dict["darp"], \
                     distb_dict["weak"], distb_dict["strong"] = train(dataLoaders["labeled"],
                                                                           dataLoaders["unlabeled"],
                                                                            model, optimizer,
                                                                            ema_optimizer,
                                                                            train_criterion,
                                                                            epoch, use_cuda,
                                                                            target_disb, emp_distb_u,
                                                                            pseudo_orig, pseudo_refine,
                                                                            lambda_u, class_weight_x, 
                                                                            class_weight_u)

        # Evaluation part
        # ___, train_acc_x, ___, train_gm_x = validate(dataLoaders["labeled"], ema_model, criterion, use_cuda, mode='Train Stats')
        test_loss, test_acc, test_cls, test_gm = validate(dataLoaders["Test"], ema_model, criterion, use_cuda, mode='Test Stats ')
        # dojoStats = dojoTest(dataLoaders, num_class, ema_model, criterion, use_cuda)

        # Use Pseudo_orig as the Class Distribution (Take Note of this!!!)
        # Note: Pseudo_refine is used for DARP (if Not, do not need)
        # print("Emperical Distribution of Unsupervised = ", emp_distb_u)
        # print("Pseudo_Original Shape = ", pseudo_orig.shape) # torch.Size([11163, 10]) # This is fine
        # print("Pseudo Refined Shape = ", pseudo_refine.shape) # torch.Size([11163, 10]) # This is fine

        print("\nFor Unlabeled Loss : ")
        class_weight_u = getClassWeights(distbLoss_dict["unlabeled"], weightLoss, epoch, args.darp, distb_dict, use_cuda)
        print("Weights_u = ", class_weight_u)

        print("For Labeled Loss : ")
        class_weight_x = getClassWeights(distbLoss_dict["labeled"], weightLoss, epoch, args.darp, distb_dict, use_cuda)
        print("Weights_x = ", class_weight_x)

        # For Next Epoch (beta)
        # if (abs(prev_train_loss / train_loss) < 1.05) :
        #     lambda_u * 1.1

        # Append logger file
        stats = [train_loss, train_loss_x, train_loss_u, 0, 0,\
            test_loss, test_acc, test_gm]
        loggerDict = appendLogger(stats, distb_dict, loggerDict, printer=True)

        is_best = False
        if test_acc > best_acc:
            best_acc = test_acc
            is_best = True

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_weight_x' : class_weight_x,
            'class_weight_u' : class_weight_u,
            'distribution' : distbLoss_dict,
            'weightLoss' : weightLoss,
        }, epoch + 1, args.out, args.save_freq, is_best=is_best)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

    closeLogger(loggerDict)

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Mean GM:')
    print(np.mean(test_gms[-20:]))

    print('Name of saved folder:')
    print(args.out)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda,
          target_disb, emp_distb_u, pseudo_orig, pseudo_refine, lambda_u, class_weight_x, class_weight_u):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    # Collect Class Distribution of Model per epoch
    output_u_all = torch.FloatTensor([]) # For Strongly Augmented
    p_hat_all = torch.FloatTensor([])   # For Weakly Augmented
    if use_cuda :
        output_u_all = output_u_all.cuda()
        p_hat_all = p_hat_all.cuda() 

    model.train()
    for batch_idx in range(args.val_iteration):

        # Prepare labeled and unlabled Batches
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            # Update the saved predictions with current one
            pseudo_orig[idx_u, :] = targets_u.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone() # torch.Size([11163, 10])

            # Applying DARP
            if args.darp and epoch > args.warm:
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig)
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    opt_res = opt_solver(pseudo_orig, target_disb)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda() 


        # Fix Match / DARP ? Still yet to be explored
        # A: Fix Match
        max_p, p_hat = torch.max(targets_u, dim=1)      # Choose Class according to Highest Prediction
        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        p_hat_all = torch.cat([p_hat_all, p_hat], dim=0) # Collect Output of Weakly Augmented Data

        # Refer to Fix Match (Supplement B2)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # Q: Why do we need inputs_u2 when select_mask is clearly repeated?
        # A: We assume delta/alpha hyperparameter = 2
        # This means we double the allowed dataset instead of clipping
        # Part of Refinement
        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0) # PyTorch Float32

        # Forward Fix Match
        # Assumption -> Ideally Supervised and Unsupervised
        # are balanced on their own right 
        # DARP tries to alleviate this issue
        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:] # Strongly Augmented Data, torch.Size([128, 10])
        output_u_all = torch.cat([output_u_all, logits_u], dim=0) # torch.Size([64000 (128*500), 10])
        # print("logits_u = ", logits_u) # Prediction numbers to be maxed out for one-hot encoding

        # SemiLoss()
        Lx, Lu = criterion(logits_x, all_targets[:batch_size], \
            logits_u, all_targets[batch_size:], select_mask, \
                class_weight_x, class_weight_u) # select_mask.size()) = torch.Size([128])

        loss = Lx + (lambda_u * Lu)  # Normally assume Regularization lambda_u = 1 (Treat Unsupervised = Supervised)
        # print("\n lambda_u * Lu_mod = ", (lambda_u * Lu))
        # print("\n loss = ", loss)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    # # Calculate Distributions:
    # Weakly Augmented Output Distribution
    weak_distb_u = torch.sum(p_hat_all, dim=0)

    # Strongly Augmented Output Distribution
    output_u_all = torch.softmax(output_u_all, dim=1)
    strong_distb_u = prob2Distribution(output_u_all, use_cuda)

    # Pseudo Label Distribution
    if use_cuda :
        pseudo_distb_u = prob2Distribution(pseudo_orig.cuda(), use_cuda) # (Non-DARP)
        darp_distb_u = prob2Distribution(pseudo_refine.cuda(), use_cuda) # DARP refined 
    else :
        pseudo_distb_u = prob2Distribution(pseudo_orig, use_cuda) # torch.Size([11163, 10]) -> torch.Size([10])
        darp_distb_u = prob2Distribution(pseudo_refine, use_cuda)
    
    return (losses.avg, losses_x.avg, losses_u.avg, emp_distb_u, \
         pseudo_orig, pseudo_refine, pseudo_distb_u, darp_distb_u, weak_distb_u, strong_distb_u)


def validate(valloader, model, criterion, use_cuda, mode):

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
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/(100 * num_class)) ** (1/num_class)
        else:
            GM *= (classwise_acc[i]) ** (1/num_class)

    return (losses.avg, classwise_acc.mean(), section_acc.cpu().numpy(), GM)

def estimate_pseudo(q_y, saved_q):
    pseudo_labels = torch.zeros(len(saved_q), num_class)
    k_probs = torch.zeros(num_class)

    for i in range(1, num_class + 1):
        i = num_class - i
        num_i = int(args.alpha * q_y[i])
        sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
        pseudo_labels[idx[: num_i], i] = 1
        k_probs[i] = sorted_probs[:num_i].sum()

    return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)

def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d

# To solve KL-Divergence Objective using Newton's Method
def opt_solver(probs, target_distb, num_iter=args.iter_T, num_newton=30):
    entropy = (-1 * probs * torch.log(probs + 1e-6)).sum(1)
    weights = (1 / entropy)
    N, K = probs.size(0), probs.size(1)

    A, w, lam, nu, r, c = probs.numpy(), weights.numpy(), np.ones(N), np.ones(K), np.ones(N), target_distb.numpy()
    A_e = A / math.e
    X = np.exp(-1 * lam / w)
    Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
    prev_Y = np.zeros(K)
    X_t, Y_t = X, Y

    for n in range(num_iter):
        # Normalization
        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom

        # Newton method
        Y_t = np.zeros(K)
        for i in range(K):
            Y_t[i] = optimize.newton(f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=1.0e-01)
        prev_Y = Y_t
        Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

    denom = np.sum(A_e * Y_t, 1)
    X_t = r / denom
    M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

    return M

# Using Library instead
# def make_imb_data(max_num, class_num, gamma):
#     mu = np.power(1/gamma, 1/(class_num - 1))
#     class_num_list = []
#     for i in range(class_num):
#         if i == (class_num - 1):
#             class_num_list.append(int(max_num / gamma))
#         else:
#             class_num_list.append(int(max_num * np.power(mu, i)))
#     print(class_num_list)
#     return list(class_num_list)

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
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        # Lu_orig = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        # print("\n Lu_orig = ", Lu_orig)

        CE_u = F.log_softmax(outputs_u, dim=1) * targets_u # Cross-Entropy Unsupervised, torch.Size([128, 10])
        WCE_u = CE_u * weights_u     # Weighted Cross-Entropy
        MCE_u = torch.sum(WCE_u, dim=1) * mask     # Masked Cross-Entropy based on Quality, torch.Size([128])
        Lu = -torch.mean(MCE_u)     # Final Unsupervised Cross-Entropy Loss
        # print("\n Lu_mod = ", Lu)

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

# def createSettings(num_class, use_cuda, \
#     distbu=args.distbu, distbl=args.distbl, \
#     invert=args.invert, normalize=args.normalize, \
#      total=args.total, minority=args.minority,  \
#          intercept=args.intercept, log=args.log, \
#              effective=args.effective, power=args.power) :
#     # Settings for Weighted loss based on Class Distribution (for Unsupervised)
#     class_weight_u = torch.ones(num_class)
#     if use_cuda :
#         class_weight_u = class_weight_u.cuda()
    
#     distbLoss_dict = {"unlabeled" : distbu, 
#                     "labeled" : distbl}

#     if (total != None and minority != None) :
#         raise Exception("Both --total and --minority declared \n \
#             Please Choose Only one of them \n \
#                 Exiting ...")
#     elif (total != None) :
#         weightLoss = {"type": "total",
#                         "const": total,
#                         "invert?" : invert,
#                         "normalize" : normalize}
#     elif (minority != None) :
#         weightLoss = {"type": "minority",
#                         "const": minority,
#                         "invert?" : invert,
#                         "normalize" : normalize}
#     elif (intercept != None) :
#       weightLoss = {"type": "intercept",
#                         "const": intercept,
#                         "invert?" : invert,
#                         "normalize" : normalize} # Feature to be Fixed
#     elif (log != None) :
#       weightLoss = {"type": "log",
#                         "const": log,
#                         "invert?" : invert,
#                         "normalize" : normalize} # Feature to be Fixed
#     elif (effective != None) :
#         weightLoss = {"type": "effective",
#                         "const": effective,
#                         "invert?" : invert,
#                         "normalize" : normalize} # Feature to be Fixed
#     elif (power != None) :
#         weightLoss = {"type": "power",
#                         "const": power,
#                         "invert?" : invert,
#                         "normalize" : normalize} # Feature to be Fixed
#     else :
#         weightLoss = None

#     return class_weight_u, distbLoss_dict, weightLoss

# def loadCheckpoint(path, model, ema_model, optimizer) :
#     out = os.path.dirname(path)
#     checkpoint = torch.load(path)
#     start_epoch = checkpoint['epoch']
#     model.load_state_dict(checkpoint['state_dict'])
#     ema_model.load_state_dict(checkpoint['ema_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
    
#     # Weighted loss based on Class Distribution (for Unsupervised)
#     class_weight_x = checkpoint['class_weight_x']
#     class_weight_u = checkpoint['class_weight_u']
#     distbLoss_dict = checkpoint['distribution']
#     weightLoss = checkpoint['weightLoss']

#     return out, start_epoch, model, ema_model, optimizer, \
#         class_weight_x, class_weight_u, distbLoss_dict, weightLoss

def printSettings(start_epoch, distbLoss_dict, weightLoss) :
    print("Starting Epoch: ", start_epoch)
    print("For Weight Loss based on Class Distribution: ")
    print("Class Distribution: ", distbLoss_dict)
    print("Weighting Formula:  ", weightLoss)

# def prob2Distribution(confidence, use_cuda):
#     '''
#     Converts Probability Output of the Model
#     into Classification Distribution through
#     Summing One-Hot Encoding
    
#     confidence : torch Array containing Probability
#                   output of the model
#     use_cuda   : Check if Using CUDA
    
#     classDistribution: torch Array containg Class Distribution
#     '''
#     maxValues, classIndex = torch.max(confidence, dim=1)
#     size = classIndex.shape[0]
#     # Transform label to one-hot
#     if use_cuda :
#         classIndex = torch.zeros(size, \
#             num_class).cuda().scatter_(1, classIndex.view(-1, 1), 1) # torch.Size([11163, 10])
#     else :
#         classIndex = torch.zeros(size, \
#             num_class).scatter_(1, classIndex.view(-1, 1), 1) # torch.Size([11163, 10])
#     classDistribution = torch.sum(classIndex, dim=0)       # torch.Size([10])

#     return classDistribution

# def invertDistribution(vec) :
#     '''
#     Returns the inverted Distribution
#     based on ascending order (inverted Tail)
#     (Class Minority -> Class Majority)

#     Inputs:
#         vec : PyTorch Tensor (Shaped as a Vector)
#               that contains the Class Distribution

#     Outputs:
#         vec : PyTorch Tensor containing Ascending
#                 Distribution
    
#     Note: Output is not strictly sorted, but arranged
#         such that the Distribution are inverted to
#         preserve the class structure
#     '''
#     values, indexes = torch.sort(vec)   # To check Ascending Distribution Values
#     L = len(vec)

#     for idx in range( ceil(L/2) ):
#         temp = deepcopy( vec[ indexes[idx] ] )
#         vec[indexes[idx]] = vec[ indexes[L-idx-1] ]
#         vec[ indexes[L-idx-1] ] = temp

#     return vec



# def prepareDataLoaders(datasets) :
#     dataLoaders = {}
#     dataLoaders["labeled"] = data.DataLoader(datasets["labeled"], batch_size=args.batch_size, shuffle=True, num_workers=4,
#                                           drop_last=True)

#     dataLoaders["unlabeled"] = data.DataLoader(datasets["unlabeled"], batch_size=args.batch_size, shuffle=True, num_workers=4,
#                                             drop_last=True)
#     dataLoaders["Test"] = data.DataLoader(datasets["Test"], batch_size=args.batch_size, shuffle=False, num_workers=4)
#     dataLoaders["Imbalanced"] = data.DataLoader(datasets["Imbalanced"], batch_size=args.batch_size, shuffle=False, num_workers=4)
#     dataLoaders["Reversed"] = data.DataLoader(datasets["Reversed"], batch_size=args.batch_size, shuffle=False, num_workers=4)
#     dataLoaders["Weak"] = data.DataLoader(datasets["Weak"], batch_size=args.batch_size, shuffle=False, num_workers=4)
#     dataLoaders["Strong"] = data.DataLoader(datasets["Strong"], batch_size=args.batch_size, shuffle=False, num_workers=4)

#     return dataLoaders

# def createLogger(path, num_class, title="") :
#     print("Creating Logger File at: ", path+'/log.txt')

#     loggerDict = {}
#     loggerDict["logger"] = Logger(os.path.join(path, 'log.txt'), title=title)
#     loggerDict["logger"].set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  \
#             'Train Acc X', 'Train GM X', \
#             'Test Loss', 'Test Acc.', 'Test GM.'])
#     loggerDict["dojo"] = Logger(os.path.join(path, 'testLog.txt'), title=title)
#     loggerDict["dojo"].set_names(['Imbalanced Loss', 'Imbalanced Acc.', 'Imbalanced GM.',  \
#             'Reversed Loss', 'Reversed Acc.', 'Reversed GM.', \
#                 'Weak Loss', 'Weak Acc.', 'Weak GM.', \
#                     'Strong Loss', 'Strong Acc.', 'Strong GM.' ])

#     loggerDict["pseudo"] = Logger(os.path.join(path, 'pseudo_distb.txt'), title=title)
#     loggerDict["pseudo"].set_names(['0-Major', '1', '2', '3', '4', '5', \
#         '6', '7', '8', '9-Minor'])
#     loggerDict["darp"] = Logger(os.path.join(path, 'darp_distb.txt'), title=title)
#     loggerDict["darp"].set_names(['0-Major', '1', '2', '3', '4', '5', \
#         '6', '7', '8', '9-Minor'])
#     loggerDict["weak"] = Logger(os.path.join(path, 'weak_distb.txt'), title=title)
#     loggerDict["weak"].set_names(['0-Major', '1', '2', '3', '4', '5', \
#         '6', '7', '8', '9-Minor'])
#     loggerDict["strong"] = Logger(os.path.join(path, 'strong_distb.txt'), title=title)
#     loggerDict["strong"].set_names(['0-Major', '1', '2', '3', '4', '5', \
#         '6', '7', '8', '9-Minor'])

#     return loggerDict

# def loadLogger(path, title="") :
#     loggerDict = {}
#     loggerDict["logger"] = Logger(os.path.join(path, 'log.txt'), title=title, resume=True)
#     loggerDict["dojo"] = Logger(os.path.join(path, 'testLog.txt'), title=title, resume=True)
#     loggerDict["pseudo"] = Logger(os.path.join(path, 'pseudo_distb.txt'), \
#         title=title+" Pseudo-Label (p) (no DARP) Distribution", resume=True)
#     loggerDict["darp"] = Logger(os.path.join(path, 'darp_distb.txt'), \
#         title=title+" Pseudo-Label (p) (w/ DARP) Distribution", resume=True)
#     loggerDict["weak"] = Logger(os.path.join(path, 'weak_distb.txt'), \
#         title=title+" Weakly Augmented Output (p_hat) Distribution", resume=True)
#     loggerDict["strong"] = Logger(os.path.join(path, 'strong_distb.txt'), \
#         title=title+" Strongly Augmented Prediction (q) Distribution", resume=True)

#     return loggerDict

# def appendLogger(stats, dojoStats, distb_dict, loggerDict, printer=False) :
#     # torch -> list
#     pseudo_distb_u = distb_dict["pseudo"].cpu().detach().tolist()
#     pseudo_distb_u = [int(p) for p in pseudo_distb_u]
#     darp_distb_u = distb_dict["darp"].detach().tolist()
#     darp_distb_u = [int(p) for p in darp_distb_u] 
#     weak_distb_u = distb_dict["weak"].cpu().detach().tolist()
#     weak_distb_u = [int(p) for p in weak_distb_u]  
#     strong_distb_u = distb_dict["strong"].cpu().detach().tolist()
#     strong_distb_u = [int(p) for p in strong_distb_u]
    
#     if printer :
#         print("Distribution (Pseudo-Label) = ", pseudo_distb_u)
#         print("Distribution (Pseudo-Label, DARPed) = ", darp_distb_u)
#         print("Weakly Augmented Distribution (Unsupervised) = ", weak_distb_u)
#         print("Strongly Augmented Distribution (Unsupervised) = ", strong_distb_u)

#     loggerDict["logger"].append(stats)
#     loggerDict["dojo"].append(dojoStats)
#     loggerDict["pseudo"].append(pseudo_distb_u)
#     loggerDict["darp"].append(darp_distb_u)
#     loggerDict["weak"].append(weak_distb_u)
#     loggerDict["strong"].append(strong_distb_u)

#     return loggerDict

# def closeLogger(loggerDict) :
#     loggerDict["logger"].close()
#     loggerDict["pseudo"].close()
#     loggerDict["darp"].close()
#     loggerDict["weak"].close()
#     loggerDict["strong"].close()

#     return loggerDict

# def gtDict(labeled, unlabeled, use_cuda) :
#     '''
#     Returns Dictionary containing Class Distributions

#     Inputs: 
#         labeled: Labeled Class Distribution (List, NumPy, PyTorch, etc)
#         unlabeled: Unlabeled Class Distribution
#     '''
#     # Collect Ground Truth of Unsupervised Class Distribution 
#     gt_distribution_l = torch.Tensor(labeled)
#     gt_distribution_u = torch.Tensor(unlabeled)
#     if use_cuda :
#         gt_distribution_l = gt_distribution_l.cuda()
#         gt_distribution_u = gt_distribution_u.cuda()
    
#     gt_distb = gt_distribution_l + gt_distribution_u

#     # Dictionary containing Class Distribution
#     distb_dict = {"gt"    : gt_distb,
#                     "gt_l"  : gt_distribution_l,
#                     "gt_u"  : gt_distribution_u} 

#     return distb_dict

# def dojoTest(dataLoaders, ema_model, criterion, use_cuda) :
#     '''
#     For Robust Testing (like in Dojo Trainings)
#     '''
#     imb_test_loss, imb_test_acc, imb_test_cls, \
#             imb_test_gm = validate(dataLoaders["Imbalanced"], ema_model, criterion, use_cuda, mode='Imbalanced Test Stats ')
#     rev_test_loss, rev_test_acc, rev_test_cls, \
#         rev_test_gm = validate(dataLoaders["Reversed"], ema_model, criterion, use_cuda, mode='Reversed Imbalanced Test Stats ')
#     weak_test_loss, weak_test_acc, weak_test_cls, \
#         weak_test_gm = validate(dataLoaders["Weak"], ema_model, criterion, use_cuda, mode='Weakly Augmented Balanced Test Stats ')
#     strong_test_loss, strong_test_acc, strong_test_cls, \
#         strong_test_gm = validate(dataLoaders["Strong"], ema_model, criterion, use_cuda, mode='Strongly Augmented Balanced Test Stats ')

#     dojoStats = [imb_test_loss, imb_test_acc, imb_test_gm, \
#         rev_test_loss, rev_test_acc, rev_test_gm, \
#             weak_test_loss, weak_test_acc, weak_test_gm, \
#                 strong_test_loss, strong_test_acc, strong_test_gm]

#     return dojoStats

if __name__ == '__main__':
    main()
