import warnings

import torch

from utils import dist
from tqdm import tqdm
import sys
import numpy as np
import torch.nn.functional as F

sys.path.append('../')
from utils import *


class EvalEngine(object):
    def __init__(self, args, val_loader, model, criterion,
                 num_class, logger=None, log_writer=None,
                 train_sample_num_per_class=None):
        self.args = args
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.num_class = num_class
        self.logger = logger
        self.log_writer = log_writer
        self.train_sample_num_per_class_labeled = train_sample_num_per_class['labeled']

        self.val_sample_num_per_class = []
        val_labels = np.array(self.val_loader.dataset.targets).astype(int)
        for i in range(num_class):
            self.val_sample_num_per_class.append(len(val_labels[val_labels == i]))

    @torch.no_grad()
    def val_one_epoch_fix(self, epoch, best_acc, best_epoch, accs=None):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # stat_meters = init_average_meters(dim=self.num_class)

        correct_num_per_class = np.zeros(self.num_class)

        self.model.eval()
        feats_all, labels_all = [], []
        for (imgs, targets, _) in tqdm(self.val_loader):
            # move data to cuda
            imgs, targets = imgs.cuda(), targets.cuda(non_blocking=True)

            # transform target to one-hot
            targets_onehot = label2onehot(targets, imgs.shape[0], self.num_class)

            feats, logits_abc, logits_base = self.model(imgs)
            feats_all.append(feats)
            labels_all.append(targets)
            # 选择用哪个分类头进行预测，默认使用ABC分类头
            logits = logits_base if self.args.eval_base else logits_abc

            scores = F.softmax(logits, dim=1)
            preds = torch.argmax(scores, dim=1)
            preds_onehot = label2onehot(preds, imgs.size()[0], self.num_class)

            loss = self.criterion(logits, targets)
            correct_num_per_class += torch.sum(targets_onehot * preds_onehot, dim=0).cpu().detach().numpy().astype(
                np.int64)

            # measure accuracy and record loss
            top1_acc, top5_acc = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(top1_acc.item(), imgs.size(0))
            top5.update(top5_acc.item(), imgs.size(0))

            #  计算pseudo-label的统计信息
            scores_base = F.softmax(logits_base, dim=1)
            scores_abc = F.softmax(logits_abc, dim=1)
            # pseudo_stat_abc = analysis_val_pseudo_labels(true_labels_L=targets, soft_pseudo_L=scores_abc,
            #                                              args=self.args, threshold=self.args.tau)
            # pseudo_stat_base = analysis_val_pseudo_labels(true_labels_L=targets, soft_pseudo_L=scores_base,
            #                                               args=self.args, threshold=self.args.tau)

            # 更新统计信息
            # stat_meters = update_average_meters(meters=stat_meters,
            #                                     stats_abc=pseudo_stat_abc,
            #                                     stats_base=pseudo_stat_base,
            #                                     n=1)

        # 当使用分布式validation的时候，需要同步不同卡之间的数据
        if self.args.distributed and self.args.dist_eval:
            correct_num_per_class = torch.tensor(correct_num_per_class, dtype=torch.float64, device='cuda')
            dist.barrier()
            dist.all_reduce(correct_num_per_class, op=torch.distributed.distributed_c10d.ReduceOp.SUM)
            correct_num_per_class = correct_num_per_class.cpu().detach().numpy()
            losses.all_reduce_between_process()
            top1.all_reduce_between_process()
            top5.all_reduce_between_process()

        ret = shot_accuracy(correct_num_per_class=correct_num_per_class,
                            many_shot_thr=self.args.many_shot_thr,
                            low_shot_thr=self.args.low_shot_thr,
                            train_num_per_class=self.train_sample_num_per_class_labeled,
                            test_num_per_class=self.val_sample_num_per_class)
        acc_per_class = ret['acc_per_class']
        all_acc, many_acc, medium_acc, low_acc = ret['all_acc'], ret['many_shot_acc'], ret['medium_shot_acc'], ret['low_shot_acc']

        half_num_class = int(self.num_class // 2)
        if acc_per_class.mean() > best_acc:
            best_epoch, best_acc = epoch + 1, acc_per_class.mean()

        # 记录日志结果
        if epoch > 30:
            last20_acc = np.mean(accs[-20:])
            last10_acc = np.mean(accs[-10:])
        else:
            last20_acc, last10_acc = 0., 0.
        if self.logger:
            self.logger.info("[EVAL@{epoch:03d}/{epochs}]loss:{loss:.4f} / "
                             "all:{all:5.2f}/head:{head:5.2f}/tail:{tail:5.2f}/"
                             "many:{many:5.2f}/med:{medium:5.2f}/low:{low:5.2f} / "
                             "best:{best_acc:5.2f}@{best_epoch} /"
                             " 10:{last10:5.2f} / 20:{last20:5.2f}",
                             epoch=epoch + 1,
                             epochs=self.args.epochs,
                             loss=losses.avg.item(),
                             all=all_acc,
                             head=acc_per_class[:half_num_class].mean(),
                             tail=acc_per_class[half_num_class:].mean(),
                             many=many_acc, medium=medium_acc, low=low_acc,
                             best_acc=best_acc, best_epoch=best_epoch,
                             last20=last20_acc, last10=last10_acc
                             )

        # log to tensorboard
        if self.log_writer:
            # global ACC
            if self.args.dataset == 'imagenet':
                self.log_writer.add_scalars(main_tag='VAL-Acc/Global', global_step=epoch,
                                            tag_scalar_dict={'all_acc': all_acc,
                                                             'many_acc': many_acc,
                                                             'medium_acc': medium_acc,
                                                             'low_acc': low_acc})
            else:
                self.log_writer.add_scalars(main_tag='VAL-Acc/Global', global_step=epoch,
                                            tag_scalar_dict={'mean_acc': acc_per_class.mean(),
                                                             'majority_acc': acc_per_class[:half_num_class].mean(),
                                                             'minority_acc': acc_per_class[half_num_class:].mean()})
            # per-class ACC
            self.log_writer.add_scalars(main_tag='VAL-Acc/Local', global_step=epoch,
                                        tag_scalar_dict={str(class_id): val.item() for class_id, val in
                                                         enumerate(acc_per_class)})

            # log pseudo statistics
            # if not self.args.disable_abc:
            #     log_stats(writer=self.log_writer, meters=stat_meters, step=epoch, mode='abc', num_class=self.num_class,
            #               split='Val', stride=self.args.writer_log_class_stride)
            # if not self.args.disable_backbone and self.args.log_backbone:
            #     log_stats(writer=self.log_writer, meters=stat_meters, step=epoch, mode='base', num_class=self.num_class,
            #               split='Val', stride=self.args.writer_log_class_stride)

        val_stat = {"mean_acc": acc_per_class.mean(),
                    "majority_acc": acc_per_class[:half_num_class].mean(),
                    'acc_per_class': acc_per_class,
                    'feats': feats_all,
                    'labels': labels_all,
                    'minority_acc': acc_per_class[half_num_class:].mean(),
                    'ret': ret}

        return val_stat
