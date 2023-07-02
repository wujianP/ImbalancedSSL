"""
Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import errno
import os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torch.optim as optim

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'set_random_seed',
           'SemiLoss', 'save_checkpoint', 'WeightEMA', 'auto_resume', 'analysis_train_pseudo_labels',
           'analysis_val_pseudo_labels', 'update_loss_average_meters', 'label2onehot',
           'generate_probs_per_class', 'get_log_unit', 'get_lr_scheduler', 'get_optimizer',
           'init_average_meters', 'update_average_meters', 'get_index', 'concat_all_gather',
           'init_pd_distribution_meters', 'update_pseudo_distribution_meters']


def get_optimizer(args, model):
    if args.optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise KeyError
    return optimizer


def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_steps, gamma=args.lr_step_gamma)
    elif args.lr_scheduler_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
    elif args.lr_scheduler_type == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(1e9))
    else:
        raise KeyError
    return scheduler


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'seed is set to :{seed}')


def auto_resume(args, logger, model, ema_model, optimizer, log_writer, scheduler):
    if args.resume:
        # Load checkpoint.
        if logger:
            logger.info(f'==> Resuming from checkpoint: {args.resume}')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        tcp_state_dict = checkpoint['tcp_state_dict']
        best_epoch = checkpoint['best_epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if log_writer:
            log_writer.set_step(checkpoint['tensorboard_step'])
        return start_epoch, tcp_state_dict, best_epoch, best_acc
    else:
        return int(0), None, int(0), 0.


def save_checkpoint(cur_acc, best_acc, best_epoch, state,
                    save_path, epoch, last_epoch=False, save_freq=1, given_save_epoch=None):
    os.makedirs(save_path, exist_ok=True)
    # save checkpoint every few epochs
    if epoch % save_freq == 0 or last_epoch:
        filepath = os.path.join(save_path, f'checkpoint@{epoch + 1}_acc{cur_acc:.2f}.pth')
        torch.save(state, filepath)
    if epoch == given_save_epoch:
        filepath = os.path.join(save_path, f'checkpoint@{epoch + 1}_acc{cur_acc:.2f}.pth')
        torch.save(state, filepath)
    # save the best checkpoint
    if cur_acc > best_acc:
        if os.path.isfile(os.path.join(save_path, f'best_acc{best_acc:.2f}@epoch{best_epoch}.pth')):
            os.remove(os.path.join(save_path, f'best_acc{best_acc:.2f}@epoch{best_epoch}.pth'))
        best_acc, best_epoch = cur_acc, epoch
        torch.save(state, os.path.join(save_path, f'best_acc{best_acc:.2f}@epoch{best_epoch}.pth'))
    # save latest checkpoint
    if os.path.isfile(os.path.join(save_path, f'latest.pth')):
        os.remove(os.path.join(save_path, f'latest.pth'))
    torch.save(state, os.path.join(save_path, f'latest.pth'))

    return best_acc, best_epoch


def label2onehot(label, batch, num_class, to_cuda=True):
    onehot = torch.zeros(batch, num_class).cpu().scatter_(1, label.view(-1, 1).cpu().long(), 1)
    return onehot.cuda() if to_cuda else onehot


def get_index(probs, get_num):
    if get_num > 0:
        try:
            probs = probs.cpu().numpy()
            probs /= probs.sum()
            index = np.random.choice(a=len(probs), size=get_num, p=probs)
            index = torch.from_numpy(index)
        except:
            index = torch.arange(get_num).long()
    else:
        index = torch.tensor([]).long()
    return index


def get_log_unit(num_class, stride):
    assert num_class % stride == 0, "num_class % stride must be divisible"
    log_unit = {}
    for i in range(int(num_class / stride)):
        start_idx = i * stride
        end_idx = start_idx + stride
        tag = f'{start_idx}-{end_idx}'
        idxs = range(start_idx, end_idx)
        log_unit[tag] = idxs
    return log_unit


def analysis_train_pseudo_labels(args, true_labels_L, soft_pseudo_U, true_labels_U=None,
                                 mask_L_for_balance=None, mask_U_for_balance=None, threshold=0.95):
    """
    Args:
        true_labels_L: true labels of labeled data, shape [B_L]
        true_labels_U: true labels of unlabeled, shape [B_U]
        soft_pseudo_U: soft_pseudo_labels of unlabeled data, shape [B_U, C]
        mask_L_for_balance: shape of [B_L], 1 means keep, 0 means masked
        mask_U_for_balance: shape of [B_U], 1 means keep, 0 means masked
        threshold: the confidence threshold of pseudo_label
        args: args of parse_args
    """
    C = soft_pseudo_U.shape[-1]  # num_classes
    B_L, B_U = args.labeled_batch_size, args.unlabeled_batch_size  # batch-size
    max_prob_U, hard_pseudo_U = torch.max(soft_pseudo_U, dim=1)  # 每个样本的置信度[B], 每个样本生成的伪标签(注意此时没有使用阈值筛选)[B]
    select_mask = max_prob_U.ge(threshold).float()  # 置信度超过阈值的样本，1表示超过, [B]
    correct_mask = (true_labels_U == hard_pseudo_U).float()  # 最大概率值对应的类与真实标签相同的样本，1表示相等, [B]
    correct_select_mask = select_mask * correct_mask  # 超过阈值且伪标签正确的样本，1表示超过且正确, [B]

    hard_onehot_U = label2onehot(label=hard_pseudo_U, batch=B_U, num_class=C)  # 把伪标签转为one-hot向量(注意此时没有使用阈值筛选), [B, C]
    gt_onehot_L = label2onehot(label=true_labels_L, batch=B_L, num_class=C)  # 把unlabeled data真实标签转为one-hot向量, [B, C]

    # 各类别的统计数据
    raw_pseudo_num_per_class = hard_onehot_U.sum(dim=0)  # 每个类取的伪标签样本数量(注意此时没有使用阈值筛选), [C]
    select_pseudo_num_per_class = select_mask @ hard_onehot_U  # 每个类别中超过阈值的伪标签数量, [B_U]x[B_U,C] = [C]
    raw_correct_num_per_class = correct_mask @ hard_onehot_U  # 每个类别中正确伪标签数量(注意此时没有使用阈值筛选), [B_U]x[B_U,C] = [C]
    select_correct_num_per_class = correct_select_mask @ hard_onehot_U  # 每个类别中超过阈值且正确伪标签数量, [B_U]x[B_U,C] = [C]
    # 每个类别伪标签正确率(注意此时没有使用阈值筛选), [C]
    raw_acc_per_class = torch.zeros(C)
    for i in range(C):
        if raw_pseudo_num_per_class[i] < 0.99:
            raw_acc_per_class[i] = 0
        else:
            raw_acc_per_class[i] = raw_correct_num_per_class[i] / raw_pseudo_num_per_class[i]
    # 每个类别超过阈值的伪标签正确率, [C]
    select_acc_per_class = torch.zeros(C)
    for i in range(C):
        if select_pseudo_num_per_class[i] < 0.99:
            select_acc_per_class[i] = 0
        else:
            select_acc_per_class[i] = select_correct_num_per_class[i] / select_pseudo_num_per_class[i]
    # 每个类别被掩码掉的样本数
    mask_L_num_per_class = (1 - mask_L_for_balance).double() @ gt_onehot_L.double()
    mask_U_num_per_class = (1 - mask_U_for_balance).double() @ hard_onehot_U.double()

    stat = {'raw_pseudo_num_per_class': raw_pseudo_num_per_class,
            'select_pseudo_num_per_class': select_pseudo_num_per_class,
            'raw_correct_num_per_class': raw_correct_num_per_class,
            'select_correct_num_per_class': select_correct_num_per_class,
            'raw_acc_per_class': raw_acc_per_class,
            'select_acc_per_class': select_acc_per_class,
            'mask_L_num_per_class': mask_L_num_per_class,
            'mask_U_num_per_class': mask_U_num_per_class}

    return stat


def analysis_val_pseudo_labels(args, true_labels_L, soft_pseudo_L, threshold=0.95):
    """
    Args:
        true_labels_L: true labels of labeled data, shape [B_L]
        soft_pseudo_L: soft_pseudo_labels of labeled data, shape [B_L, C]
        threshold: the confidence threshold of pseudo_label
        args: args of parse_args
    """
    C = soft_pseudo_L.shape[-1]  # num_classes
    B = args.val_batch_size  # batch-size
    max_prob_L, hard_pseudo_L = torch.max(soft_pseudo_L, dim=1)  # 每个样本的置信度[B_L], 每个样本生成的伪标签(注意此时没有使用阈值筛选)[B_L]

    select_mask = max_prob_L.ge(threshold).float()  # 置信度超过阈值的样本，1表示超过, [B_L]
    correct_mask = (true_labels_L == hard_pseudo_L).float()  # 最大概率值对应的类与真实标签相同的样本，1表示相等, [B_L]
    correct_select_mask = select_mask * correct_mask  # 超过阈值且伪标签正确的样本，1表示超过且正确, [B_L]

    hard_onehot_L = label2onehot(hard_pseudo_L, B, C)  # 把伪标签转为one-hot向量(注意此时没有使用阈值筛选), [B_L, C]

    # 各类别的统计数据
    raw_pseudo_num_per_class = hard_onehot_L.sum(dim=0)  # 每个类取的伪标签样本数量(注意此时没有使用阈值筛选), [C]
    select_pseudo_num_per_class = select_mask @ hard_onehot_L  # 每个类别中超过阈值的伪标签数量, [B_L]x[B_L,C] = [C]
    raw_correct_num_per_class = correct_mask @ hard_onehot_L  # 每个类别中正确伪标签数量(注意此时没有使用阈值筛选), [B_L]x[B_L,C] = [C]
    select_correct_num_per_class = correct_select_mask @ hard_onehot_L  # 每个类别中超过阈值且正确伪标签数量, [B_L]x[B_L,C] = [C]
    # 每个类别伪标签正确率(注意此时没有使用阈值筛选), [C]
    raw_acc_per_class = torch.zeros(C)
    for i in range(C):
        if raw_pseudo_num_per_class[i] < 0.99:
            raw_acc_per_class[i] = 0
        else:
            raw_acc_per_class[i] = raw_correct_num_per_class[i] / raw_pseudo_num_per_class[i]
    # 每个类别超过阈值的伪标签正确率, [C]
    select_acc_per_class = torch.zeros(C)
    for i in range(C):
        if select_pseudo_num_per_class[i] < 0.99:
            select_acc_per_class[i] = 0
        else:
            select_acc_per_class[i] = select_correct_num_per_class[i] / select_pseudo_num_per_class[i]
    # 每个类别被掩码掉的样本数
    mask_L_num_per_class = torch.zeros(C)
    mask_U_num_per_class = torch.zeros(C)

    stat = {'raw_pseudo_num_per_class': raw_pseudo_num_per_class,
            'select_pseudo_num_per_class': select_pseudo_num_per_class,
            'raw_correct_num_per_class': raw_correct_num_per_class,
            'select_correct_num_per_class': select_correct_num_per_class,
            'raw_acc_per_class': raw_acc_per_class,
            'select_acc_per_class': select_acc_per_class,
            'mask_L_num_per_class': mask_L_num_per_class,
            'mask_U_num_per_class': mask_U_num_per_class}

    return stat


def init_average_meters(dim):
    meters = {
        'raw_pseudo_num_per_class_base': AverageMeter(dim=dim),
        'select_pseudo_num_per_class_base': AverageMeter(dim=dim),
        'raw_correct_num_per_class_base': AverageMeter(dim=dim),
        'select_correct_num_per_class_base': AverageMeter(dim=dim),
        'raw_acc_per_class_base': AverageMeter(dim=dim),
        'select_acc_per_class_base': AverageMeter(dim=dim),
        'raw_pseudo_num_per_class_abc': AverageMeter(dim=dim),
        'select_pseudo_num_per_class_abc': AverageMeter(dim=dim),
        'raw_correct_num_per_class_abc': AverageMeter(dim=dim),
        'select_correct_num_per_class_abc': AverageMeter(dim=dim),
        'raw_acc_per_class_abc': AverageMeter(dim=dim),
        'select_acc_per_class_abc': AverageMeter(dim=dim),
        'mask_L_num_per_class_base': AverageMeter(dim=dim),
        'mask_U_num_per_class_base': AverageMeter(dim=dim),
        'mask_L_num_per_class_abc': AverageMeter(dim=dim),
        'mask_U_num_per_class_abc': AverageMeter(dim=dim),
        # loss meters
        'losses_total': AverageMeter(),
        'losses_L_base': AverageMeter(),
        'losses_U_base': AverageMeter(),
        'losses_L_abc': AverageMeter(),
        'losses_U_abc': AverageMeter(),
        'losses_tcp': AverageMeter(),
        # data meters
        'data_time': AverageMeter(),
        'batch_time': AverageMeter(),
        'writer_time': AverageMeter(),
    }
    return meters


def update_average_meters(meters, stats_base, stats_abc, n=1):
    keys = ['raw_pseudo_num_per_class', 'select_pseudo_num_per_class', 'raw_correct_num_per_class',
            'select_correct_num_per_class', 'raw_acc_per_class', 'select_acc_per_class', 'mask_L_num_per_class',
            'mask_U_num_per_class']
    for key in keys:
        if stats_base:
            meters[f"{key}_base"].update(stats_base[key].cpu(), n)
        if stats_abc:
            meters[f"{key}_abc"].update(stats_abc[key].cpu(), n)
    return meters


def update_pseudo_distribution_meters(meters, stats_base, stats_abc, ema_momentum_wt, args, n=1):
    keys = ['raw_pseudo_num_per_class', 'select_pseudo_num_per_class']
    # use every n epoch estimate
    if args.pd_distribution_estimate_nepoch > 0:
        for key in keys:
            if stats_base:
                meters[f"{key}_base"].update(val=stats_base[key].cpu(), n=n)
            if stats_abc:
                meters[f"{key}_abc"].update(val=stats_abc[key].cpu(), n=n)
    # use ema estimate
    elif args.pd_stat_ema_momentum_wt > 0:
        for key in keys:
            if stats_base:
                meters[f"{key}_base"].ema_update(val=stats_base[key].cpu(), ema_momentum=ema_momentum_wt, n=n)
            if stats_abc:
                meters[f"{key}_abc"].ema_update(val=stats_abc[key].cpu(), ema_momentum=ema_momentum_wt, n=n)
    else:
        raise KeyError
    return meters


def update_loss_average_meters(meters, losses, n=1):
    for key, value in losses.items():
        meters[key].update(value, n)
    return meters


def init_pd_distribution_meters(dim):
    meters = {
        'raw_pseudo_num_per_class_base': AverageMeter(dim=dim),
        'select_pseudo_num_per_class_base': AverageMeter(dim=dim),
        'raw_pseudo_num_per_class_abc': AverageMeter(dim=dim),
        'select_pseudo_num_per_class_abc': AverageMeter(dim=dim)
    }
    return meters


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def generate_probs_per_class(sample_fun_type, norm_type, stat_per_class, power=1, to_cuda=True):
    # 当Nk=0时，设置为Nk=1，解决除零错
    stat_per_class = torch.clamp(input=stat_per_class, min=0.001)

    # 计算probs_distribution
    if sample_fun_type == 'exp_min':
        stat_per_class = stat_per_class.float() / power
        probs = stat_per_class.min().exp() / stat_per_class.exp()
    elif sample_fun_type == 'poly_inv':
        probs = 1 / stat_per_class.float()
        probs = probs.pow(power)
    elif sample_fun_type == 'abc':
        probs = stat_per_class.min() / stat_per_class.float()
        norm_type = 'no'
    else:
        raise KeyError

    # norm to one
    if norm_type == 'global':
        probs = probs / probs.sum()
    elif norm_type == 'local':
        probs = probs / probs.max()
    elif norm_type == 'no':
        probs = probs
    else:
        raise KeyError

    if to_cuda:
        probs = probs.cuda()

    probs = probs.nan_to_num(nan=0.)

    return probs


class SemiLoss(object):
    def __call__(self,
                 logits_L,
                 targets_L,
                 logits_U,
                 targets_U,
                 mask_L_for_balance,
                 mask_U_for_balance,
                 mask_U_by_threshold):
        """
        Args:
            logits_L:
            targets_L:
            logits_U:
            targets_U:
            mask_L_for_balance:
            mask_U_for_balance:
            mask_U_by_threshold:
        """
        loss_L = -torch.mean(torch.sum(F.log_softmax(logits_L, dim=1) * targets_L, dim=1) * mask_L_for_balance)
        loss_U = -torch.mean(
            torch.sum(F.log_softmax(logits_U, dim=1) * targets_U, dim=1) * mask_U_by_threshold * mask_U_for_balance)

        return loss_L, loss_U


class WeightEMA(object):
    def __init__(self,
                 model,
                 ema_model,
                 lr,
                 alpha=0.999,
                 do_wd=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        if do_wd:
            self.wd = 0.02 * lr
        else:
            self.wd = 0.

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param, param = ema_param.float(), param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, dim=1):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.dim = dim
        self.reset()

    def reset(self):
        self.val = torch.zeros(self.dim)
        self.avg = torch.zeros(self.dim)
        self.sum = torch.zeros(self.dim)
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def ema_update(self, val, ema_momentum, n=1):
        self.val = ema_momentum * self.val + (1 - ema_momentum) * val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce_between_process(self):
        from utils import dist
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        t = torch.tensor([self.count, self.sum, self.avg, self.val],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, op=torch.distributed.distributed_c10d.ReduceOp.AVG)
        self.count, self.sum, self.avg, self.val = int(t[0].item()), t[1], t[2], t[3]
