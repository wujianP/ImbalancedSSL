from utils import *

import argparse
import math
import time
import os
import logging
import shutil
from tqdm import tqdm
import random

import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets.models.ema import WeightEMA
from datasets.models.wideresnet import WideResNet
from algorithms.fixmatch import FixMatch, ADSH
from datasets.load_imb_data import *
from utils.misc import *
from config import config

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pytorch SSL Library')
parser.add_argument('--gpu-ids', default='0', type=str, help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--alg', default='adsh', type=str, choices=['supervised', 'FM', 'adsh'],help='algorithms')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
parser.add_argument('--num_classes', type=int, default=127,
                    help='number of classes')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total steps to run')
parser.add_argument('--eval_steps', default=500, type=int,
                    help='number of eval steps to run per epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--lambda_u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--mu', default=1, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='threshold for pseduo-label confidence')
parser.add_argument('--optim', default='ADAM', type=str,
                    choices=['SGD', 'ADAM'])
parser.add_argument('--lr', default=0.002, type=float,
                    metavar='LR', help='2e-3 for ADAM and 3e-2 for SGD')
parser.add_argument('--weight_decay', default=0.0005, type=float,
                    help='weight decay of SGD')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='use nesterov')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--use_ema',  default=True, type=bool,
                    help='use EMA model')
parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--out', default='result_imb',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# ADD by WJ
parser.add_argument('--labeled_percent', type=float, default=0.1, help='by default we take 10% labeled data')
parser.add_argument('--img_size', type=int, default=32, help='ImageNet127_32 or ImageNet127_64')
parser.add_argument('--save_freq', type=int, default=10)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--find_unused_parameters', action='store_true')

args = parser.parse_args()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(args.manualSeed)


def save_checkpoint(state, epoch, save_path, save_freq, is_best):
    if epoch % save_freq == 0:
        torch.save(state, os.path.join(save_path, f'checkpoint_{epoch}.pth'))
    # save the best checkpoint
    if is_best:
        if os.path.isfile(os.path.join(save_path, 'best.pth')):
            os.remove(os.path.join(save_path, 'best.pth'))
        torch.save(state, os.path.join(save_path, 'best.pth'))


def build_model(args, ema=False):
    from datasets.models.resnet import ResNet50
    model = ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)
    model = model.cuda()

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj, result_logger):

    model = build_model(args)
    ema_model = build_model(args, ema=True)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    else:
        raise KeyError

    ema_optimizer = WeightEMA(model_without_ddp, ema_model, ema_decay=args.ema_decay, lr=args.lr)
    optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr)

    args.start_epoch = 0

    conf = [[] for _ in range(args.num_classes)]
    conf_std = [[] for _ in range(args.num_classes)]
    if args.resume:
        if is_main_process():
            logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if is_main_process():
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total optimization steps = {args.epochs * args.eval_steps}")

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    labeled_iter = iter(label_loader)
    unlabeled_iter = iter(unlabel_loader)
    score = np.zeros(args.num_classes) + args.threshold

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            label_loader.sampler.set_epoch(epoch)
            unlabel_loader.sampler.set_epoch(epoch)

        # FIXME
        # if args.alg == 'adsh' and epoch > 1:
        #     score = update_s(args, score, unlabel_loader, model)

        lists = [[] for _ in range(args.num_classes)]

        model.train()
        p_bar = tqdm(range(args.eval_steps))
        for batch_idx in p_bar:
            try:
                inputs_l, targets = labeled_iter.next()
            except:
                labeled_iter = iter(label_loader)
                inputs_l, targets = labeled_iter.next()

            try:
                (inputs_u, _) = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabel_loader)
                (inputs_u, _) = unlabeled_iter.next()

            data_time.update(time.time() - end)

            inputs_l = inputs_l.cuda()
            targets = targets.cuda().long()
            inputs_uw, inputs_us = inputs_u[0].cuda(), inputs_u[1].cuda()
            # concate
            inputs_all = torch.cat([inputs_l, inputs_uw, inputs_us])
            logits_all = model(inputs_all)[0]

            logits = logits_all[:args.batch_size]
            outputs_uw = logits[args.batch_size: args.batch_size * 2]
            outputs = logits[args.batch_size * 2:]

            cls_loss = F.cross_entropy(logits, targets)

            # outputs_uw = model(inputs_uw)[0]
            probs = torch.softmax(outputs_uw, dim=1)
            rectify_prob = probs / torch.from_numpy(score).float().cuda()
            max_rp, rp_hat = torch.max(rectify_prob, dim=1)
            mask = max_rp.ge(1.0)

            # outputs = model(inputs_us)[0]

            ssl_loss = (F.cross_entropy(outputs, rp_hat, reduction='none') * mask).mean()

            loss = cls_loss + args.lambda_u * ssl_loss

            losses.update(loss.item(), inputs_l.size(0))
            losses_x.update(cls_loss.item(), inputs_l.size(0))
            losses_u.update(ssl_loss.item(), inputs_l.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            # for score update
            logits = torch.softmax(outputs_uw.detach(), dim=1)
            max_probs, targets_u = torch.max(logits, dim=1)
            for i in range(targets_u.shape[0]):
                lists[targets_u[i]].append(max_probs[i].detach().cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if is_main_process():
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:}. Iter: {batch:}/{iter:}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_steps,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg))

            p_bar.update()

        if args.alg == 'adsh' and epoch > 1:
            lists = np.array(lists, dtype=object)
            rho = 1.0
            for i in range(lists.shape[0]):
                lists[i] = np.sort(np.array(lists[i]))[::-1]
            for i in range(lists[0].shape[0]):
                if lists[0][i] < args.threshold:
                    break
                rho = (i + 1) / lists[0].shape[0]
            for i in range(1, lists.shape[0]):
                if lists[i].shape[0] != 0:
                    idx = max(0, np.round(lists[i].shape[0] * rho - 1).astype(int))
                    score[i] = min(args.threshold, lists[i][idx])

        p_bar.close()

        test_loss, test_acc = test(args, test_loader, ema_model)

        is_best = False
        if test_acc > best_acc:
            best_acc = test_acc
            is_best = True
            if is_main_process():
                print(f'BEST:!!!!!!!!!!!!!{best_acc}')

        if is_main_process():
            result_logger.append([0, 0, 0, 0, test_loss, test_acc])
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_without_ddp.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, args.out, save_freq=args.save_freq, is_best=is_best)

        test_accs.append(test_acc)
        if is_main_process():
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(test_acc))


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader)
    classwise_correct = torch.zeros(args.num_classes).cuda()
    classwise_num = torch.zeros(args.num_classes).cuda()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            y_true.extend(targets.tolist())

            inputs = inputs.cuda()
            targets = targets.cuda().long()
            outputs = model(inputs)[0]
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            # classwise prediction
            pred_label = outputs.max(1)[1]
            y_pred.extend(pred_label.tolist())
            pred_mask = (targets == pred_label).float()
            for i in range(args.num_classes):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

            batch_time.update(time.time() - end)
            end = time.time()
            if is_main_process():
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        test_loader.close()

    classwise_acc = (classwise_correct / classwise_num)
    if is_main_process():
        logger.info("top-1 acc: {:.2f}".format(top1.avg))
        logger.info("top-5 acc: {:.2f}".format(top5.avg))

    return losses.avg, classwise_acc.mean()


def main():
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # distributed config
    init_distributed_mode(args)

    if is_main_process():
        from utils.logger import Logger
        result_logger = Logger(os.path.join(args.out, 'result_log.txt'), title='fix-ImageNet-127')
        result_logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss Teacher', 'Test Loss', 'Test Acc.'])

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)

        logger.info(dict(args._get_kwargs()))
    else:
        result_logger = None

    if is_main_process():
        print(f'==> Preparing small ImageNet127')
    from datasets.fix_small_imagenet127 import get_small_imagenet
    img_size2path = {32: '/dev/shm/small_imagenet127/res32', 64: '/dev/shm/small_imagenet127/res64'}
    tmp = get_small_imagenet(img_size2path[args.img_size], args.img_size, labeled_percent=args.labeled_percent,
                             seed=args.manualSeed, return_strong_labeled_set=False)

    N_SAMPLES_PER_CLASS, train_labeled_set, train_unlabeled_set, test_set = tmp

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        labeled_train_sampler = torch.utils.data.DistributedSampler(
            dataset=train_labeled_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        unlabeled_train_sampler = torch.utils.data.DistributedSampler(
            dataset=train_unlabeled_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        raise KeyError

    label_loader = torch.utils.data.DataLoader(train_labeled_set,
                                               batch_size=args.batch_size,
                                               sampler=labeled_train_sampler,
                                               num_workers=8,
                                               drop_last=True)
    unlabel_loader = torch.utils.data.DataLoader(train_unlabeled_set,
                                                 batch_size=args.batch_size,
                                                 sampler=unlabeled_train_sampler,
                                                 num_workers=8,
                                                 drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if args.alg == 'supervised':
        raise KeyError
    elif args.alg == 'FM':
        raise KeyError
    elif args.alg == 'adsh':
        ssl_obj = ADSH(args, 1, args.threshold)
        train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj, result_logger=result_logger)


if __name__ == '__main__':
    main()
