# This code is constructed based on Pytorch Implementation of MixMatch(https://github.com/YU1ut/MixMatch-pytorch)

from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import models.wrn as models
import dataset.remix_cifar10 as dataset_cifar10
import dataset.remix_cifar100 as dataset_cifar100
from utils import make_imb_data, save_checkpoint, ReMixMatch_Loss, WeightEMA, linear_rampup, interleave
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
# Method options
parser.add_argument('--width', type=int, default=2, help='wrn width')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--num_max', type=int, default=1500, help='Number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0, help='Relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio_l', type=int, default=100, help='Imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100, help='Imbalance ratio for unlabeled data')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')
# Hyperparameters for ReMixMatch
parser.add_argument('--mix_alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=1.5, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--w_rot', default=0.5, type=float)
parser.add_argument('--w_ent', default=0.5, type=float)
parser.add_argument('--align', action='store_true', help='Distribution alignment term')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
cudnn.benchmark = False
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True

best_acc = 0  # best test accuracy
if args.dataset == 'cifar10':
    num_class = 10
elif args.dataset == 'cifar100':
    num_class = 100
else:
    raise NotImplementedError


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced {args.dataset}')

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio_u)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, test_set = dataset_cifar10.get_cifar10('/BS/databases00/cifar-10',
                                                                                       N_SAMPLES_PER_CLASS,
                                                                                       U_SAMPLES_PER_CLASS, seed=args.manualSeed)
    elif args.dataset == 'cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset_cifar100.get_cifar100('/BS/databases00/cifar-100',
                                                                                         N_SAMPLES_PER_CLASS,
                                                                                         U_SAMPLES_PER_CLASS, seed=args.manualSeed)
    else:
        raise NotImplementedError

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                            drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(args.width, num_class)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = ReMixMatch_Loss(args.lambda_u, args.epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, args.lr, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'remix-cifar-10'
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
        emp_distb_u = checkpoint['emp_distb_u']
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Total Acc.', 'Test Loss', 'Test Acc.', 'Test GM.'])
        emp_distb_u = torch.ones(num_class) / num_class
    test_accs = []
    test_gms = []

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        target_disb = N_SAMPLES_PER_CLASS_T * sum(U_SAMPLES_PER_CLASS) / sum(N_SAMPLES_PER_CLASS)
        # Note: for this implementation, whether normalize target_disb leads to equivalent result!!!

        # Training part
        train_loss, train_loss_x, train_loss_u, total_c, emp_distb_u = train(labeled_trainloader,
                                                                             unlabeled_trainloader,
                                                                             model, optimizer,
                                                                             ema_optimizer,
                                                                             train_criterion,
                                                                             epoch, use_cuda,
                                                                             target_disb, emp_distb_u)

        # Evaluation part
        test_loss, test_acc, test_cls, test_gm = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, total_c, test_loss, test_acc, test_gm])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'emp_distb_u': emp_distb_u,
        }, epoch, args.out)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Mean GM:')
    print(np.mean(test_gms[-20:]))

    print('Name of saved folder:')
    print(args.out)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda,
          target_disb, emp_distb_u):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_r = AverageMeter()
    losses_e = AverageMeter()
    total_c = AverageMeter()
    ws = AverageMeter()
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
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Rotate images
        temp = []
        targets_r = torch.randint(0, 4, (inputs_u2.size(0),)).long()
        for i in range(inputs_u2.size(0)):
            inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 32, 32)
            temp.append(inputs_rot)
        inputs_r = torch.cat(temp, 0)
        targets_r = torch.zeros(batch_size, 4).scatter_(1, targets_r.view(-1, 1), 1)
        inputs_r, targets_r = inputs_r.cuda(), targets_r.cuda(non_blocking=True)

        # Generate the pseudo labels
        with torch.no_grad():
            outputs_u, _ = model(inputs_u)
            p = torch.softmax(outputs_u, dim=1)

            # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
            real_batch_idx = batch_idx + epoch * args.val_iteration
            if real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            # Distribution alignment
            if args.align:
                pa = p * (target_disb.cuda() + 1e-6) / (emp_distb_u.mean(0).cuda() + 1e-6)
                p = pa / pa.sum(dim=1, keepdim=True)

            # Temperature scailing
            pt = p ** (1 / args.T)
            targets_u = (pt / pt.sum(dim=1, keepdim=True)).detach()

            max_p, p_hat = torch.max(targets_u, dim=1)
            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            total_c.update(total_acc.mean(0).item())

        # Mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

        l = np.random.beta(args.mix_alpha, args.mix_alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        _, logits_r = model(inputs_r)
        Lr = -1 * torch.mean(torch.sum(F.log_softmax(logits_r, dim=1) * targets_r, dim=1))

        # Entropy minimization for unlabeled samples (strong augmented)
        outputs_u2, _ = model(inputs_u2)
        Le = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u, dim=1))
        loss = Lx + w * Lu + args.w_rot * Lr + args.w_ent * Le * linear_rampup(epoch+batch_idx/args.val_iteration, args.epochs)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_r.update(Lr.item(), inputs_x.size(0))
        losses_e.update(Le.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_r: {loss_r:.4f} | ' \
                      'Loss_e: {loss_e:.4f} | Total_acc: {total_c:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_r=losses_r.avg,
                    loss_e=losses_e.avg,
                    total_c=total_c.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, total_c.avg, emp_distb_u)


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

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            y_true.extend(targets.tolist())

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
            pred_label = outputs.max(1)[1]
            y_pred.extend(pred_label.tolist())
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
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

    return (losses.avg, top1.avg, section_acc.cpu().numpy(), GM)


if __name__ == '__main__':
    main()
