# This code is constructed based on Pytorch Implementation of MixMatch(https://github.com/YU1ut/MixMatch-pytorch)

from __future__ import print_function

import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import argparse
import time
from dataset.fix_imagenet import get_imagenet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel

from utils import Bar, Logger, AverageMeter, mkdir_p
from common import validate, estimate_pseudo, opt_solver, save_checkpoint, SemiLoss, \
    WeightEMA

parser = argparse.ArgumentParser(description='PyTorch DARP Training')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# Common Hyper-parameters for Semi-supervised Methods
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.7, type=float, help='hyper-parameter for pseudo-label of FixMatch')

# Hyperparameters for DARP
parser.add_argument('--warm', type=int, default=120, help='Number of warm up epoch for DARP')
parser.add_argument('--alpha', default=2.0, type=float, help='hyperparameter for removing noisy entries')
parser.add_argument('--darp', action='store_true', help='Applying DARP')
parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')

# added by wj
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--annotation_file_path', required=True, type=str)
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--labeled_ratio', type=int)

parser.add_argument('--max_num_l', type=int, default=600)
parser.add_argument('--max_num_u', type=int, default=300)
parser.add_argument('--imb_ratio_l', type=int)
parser.add_argument('--imb_ratio_u', type=int)

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
num_class = 1000
args.num_class = 1000


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced ImageNetLT-{args.labeled_ratio}')
    tmp = get_imagenet(
        root=args.data_path,
        annotation_file_train_labeled=f'{args.annotation_file_path}/maxL{args.max_num_l}_maxU{args.max_num_u}_imbL{args.imb_ratio_l}_imbU{args.imb_ratio_u}_labeled.txt',
        annotation_file_train_unlabeled=f'{args.annotation_file_path}/maxL{args.max_num_l}_maxU{args.max_num_u}_imbL{args.imb_ratio_l}_imbU{args.imb_ratio_u}_unlabeled.txt',
        annotation_file_val=f'{args.annotation_file_path}/ImageNet_LT_val.txt',
        num_per_class=f'{args.annotation_file_path}/maxL{args.max_num_l}_maxU{args.max_num_u}_imbL{args.imb_ratio_l}_imbU{args.imb_ratio_u}_sampleNum.txt')
    sample_num_per_class, train_labeled_set, train_unlabeled_set, test_set = tmp

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Data
    N_SAMPLES_PER_CLASS = sample_num_per_class['labeled']
    U_SAMPLES_PER_CLASS = sample_num_per_class['unlabeled']
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    # Model
    print("==> creating ResNet50")

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

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = f'Imbalanced-ImageNetLT-labeled-ratio-{args.labeled_ratio}'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc.', 'Test GM.'])

    test_accs = []

    # Default values for MixMatch and DARP
    emp_distb_u = torch.ones(args.num_class) / args.num_class
    pseudo_orig = torch.ones(len(train_unlabeled_set), args.num_class) / args.num_class
    pseudo_refine = torch.ones(len(train_unlabeled_set), args.num_class) / args.num_class

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Use the estimated distribution of unlabeled data
        if args.est:
            raise KeyError
        # Use the inferred distribution with labeled data
        else:
            target_disb = N_SAMPLES_PER_CLASS_T * len(train_unlabeled_set) / sum(N_SAMPLES_PER_CLASS)
        
        train_loss, train_loss_x, train_loss_u, emp_distb_u, pseudo_orig, pseudo_refine = train_fix(args, labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch, use_cuda,
                                                                                                target_disb, emp_distb_u,
                                                                                                pseudo_orig, pseudo_refine)

        # Evaluation part
        test_loss, test_acc, test_cls, class_wise_acc = validate(test_loader, ema_model, criterion, use_cuda,
                                                          mode='Test Stats', num_class=args.num_class)

        is_best = False
        test_acc = class_wise_acc.mean()
        if test_acc >= best_acc:
            best_acc = test_acc
            is_best = True

        # Append logger file
        print(f'Epoch:{epoch}-Acc{test_acc}')
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc, 0.])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, args.out, save_freq=args.save_freq, is_best=is_best)
        test_accs.append(test_acc)

    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))


def train_fix(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
              use_cuda, target_disb, emp_distb_u, pseudo_orig, pseudo_refine):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
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
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            # Update the saved predictions with current one
            pseudo_orig[idx_u, :] = targets_u.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone()

            # Applying DARP
            if args.darp and epoch > args.warm:
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig, args.num_class, args.alpha)
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.1)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda()

        max_p, p_hat = torch.max(targets_u, dim=1)
        p_hat = torch.zeros(batch_size, args.num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)

        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(args, logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], epoch, select_mask)
        loss = Lx + Lu

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
    return (losses.avg, losses_x.avg, losses_u.avg, emp_distb_u, pseudo_orig, pseudo_refine)


if __name__ == '__main__':
    main()
