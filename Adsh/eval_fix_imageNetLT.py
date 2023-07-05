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

from sklearn.metrics import precision_score, recall_score
from datasets.models.ema import WeightEMA
from datasets.models.wideresnet import WideResNet
from algorithms.fixmatch import FixMatch, ADSH
from datasets.load_imb_data import *
from utils.misc import *
from config import config

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pytorch SSL Library')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--alg', default='adsh', type=str, choices=['supervised', 'FM', 'adsh'],help='algorithms')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
parser.add_argument('--num_classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument('--total_steps', default=150000, type=int,
                    help='number of total steps to run')
parser.add_argument('--eval_steps', default=500, type=int,
                    help='number of eval steps to run per epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--lambda_u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--mu', default=1, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.7, type=float,
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
parser.add_argument('--labeled_ratio', type=float, default=20, help='by default we take 20% labeled data')
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--annotation_file_path', required=True, type=str)
parser.add_argument('--save_freq', type=int, default=30)


parser.add_argument('--many_shot_thr', type=int)
parser.add_argument('--low_shot_thr', type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    from datasets.models.resnetwithABC import ResNet
    model = ResNet(num_classes=args.num_classes, encoder_name='resnet', pretrained=False)
    model = model.cuda()

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def update_s(args, score, unlabel_loader, model):
    with torch.no_grad():
        lists = [[] for _ in range(args.num_classes)]
        for batch_idx, ((inputs_uw, inputs_us), _) in enumerate(unlabel_loader):
            model.eval()
            inputs_uw = inputs_uw.cuda()
            outputs = model(inputs_uw)[0]
            logits = torch.softmax(outputs.detach(), dim=1)
            max_probs, targets_u = torch.max(logits, dim=1)

            for i in range(targets_u.shape[0]):
                lists[targets_u[i]].append(max_probs[i].detach().cpu().numpy())

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
        return score


def train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj, result_logger, sample_num_train):
    args.epochs = math.ceil(args.total_steps / args.eval_steps)

    model = build_model(args)
    ema_model = build_model(args, ema=True)
    ema_optimizer = WeightEMA(model, ema_model, ema_decay=args.ema_decay, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.start_epoch = 0

    conf = [[] for _ in range(args.num_classes)]
    conf_std = [[] for _ in range(args.num_classes)]
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    test_loss, test_acc = test(args, test_loader, ema_model, sample_num_train=sample_num_train)


def test(args, test_loader, model, sample_num_train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader)

    classwise_correct = torch.zeros(1000).cuda()
    classwise_num = torch.zeros(1000).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.cuda()
            targets = targets.cuda().long()
            outputs = model(inputs)[0]
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            # classwise prediction
            pred_label = outputs.max(1)[1]  # torch.Size([16])
            pred_mask = (targets == pred_label).float()  # torch.Size([16])

            for i in range(1000):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

            batch_time.update(time.time() - end)
            end = time.time()
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

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


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
        'many_shot_acc': np.mean(many_shot_acc),
        'medium_shot_acc': np.mean(median_shot_acc),
        'low_shot_acc': np.mean(low_shot_acc)
    }

    return ret


def main():
    from utils.logger import Logger
    result_logger = Logger(os.path.join(args.out, 'result_log.txt'), title='fix-ImageNet-LT')
    result_logger.set_names(
        ['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss Teacher', 'Test Loss', 'Test Acc.'])
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))

    print(f'==> Preparing imbalanced ImageNetLT-{args.labeled_ratio}')
    from datasets.fix_imagenet import get_imagenet
    tmp = get_imagenet(
        root=args.data_path,
        annotation_file_train_labeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_labeled.txt',
        annotation_file_train_unlabeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_unlabeled.txt',
        annotation_file_val=f'{args.annotation_file_path}/ImageNet_LT_test.txt',
        num_per_class=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_sample_num.txt')
    sample_num_PPP, train_labeled_set, train_unlabeled_set, test_set = tmp

    sample_num_train = sample_num_PPP['labeled']

    label_loader = torch.utils.data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                          drop_last=True)
    unlabel_loader = torch.utils.data.DataLoader(train_unlabeled_set, batch_size=args.mu * args.batch_size, shuffle=True,
                                            num_workers=6, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=6)

    if args.alg == 'supervised':
        train_ssl(label_loader, unlabel_loader, test_loader, None, result_logger=result_logger)
    elif args.alg == 'FM':
        ssl_obj = FixMatch(args, 1, args.threshold)
        train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj, result_logger=result_logger)
    elif args.alg == 'adsh':
        ssl_obj = ADSH(args, 1, args.threshold)
        train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj, result_logger=result_logger, sample_num_train=sample_num_train)


if __name__ == '__main__':
    main()
