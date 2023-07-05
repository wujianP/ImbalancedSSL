import torch
from tensorboardX import SummaryWriter
from loguru import logger
from .misc import get_log_unit

__all__ = ['TensorboardLogger', 'create_logger', 'log_loss', 'log_stats', 'log_single_stat']


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


def create_logger(args, remove_console=False):

    log_writer = TensorboardLogger(log_dir=f"{args.out}/tensorboard")

    logger.add(args.out + "/log/config_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "config_log")
    logger.add(args.out + "/log/train_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "train_log")
    logger.add(args.out + "/log/val_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "val_log")
    logger.add(args.out + "/log/tcp_num_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "tcp_num_log")
    logger.add(args.out + "/log/tcp_acc_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "tcp_acc_log")
    logger.add(args.out + "/log/tcp_get_num_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "tcp_get_num_log")
    logger.add(args.out + "/log/mis_log@{time:MM-DD_HH:mm}.log", format="{message}", level="INFO",
               filter=lambda record: record["extra"]["name"] == "mis_log")

    if remove_console:
        logger.remove(handler_id=None)

    cfg_logger = logger.bind(name='config_log')
    train_logger = logger.bind(name='train_log')
    val_logger = logger.bind(name='val_log')
    tcp_num_logger = logger.bind(name='tcp_num_log')
    tcp_acc_logger = logger.bind(name='tcp_acc_log')
    tcp_get_num_logger = logger.bind(name='tcp_get_num_log')
    mis_logger = logger.bind(name='mis_log')

    return log_writer, cfg_logger, train_logger, val_logger, tcp_num_logger, tcp_acc_logger, tcp_get_num_logger, mis_logger


def log_stats(writer, meters, mode, num_class, split, stride, step=-1):
    """
    mode: abc or base
    num_class: number of class
    split: train or val
    stride: log stride of all class
    step: step of log_writer
    """
    assert mode in ['base', 'abc']
    assert split in ['Train', 'Val']
    step = writer.step if step == -1 else step
    # 生成以stride为间隔的记录单元
    log_unit = get_log_unit(num_class=num_class, stride=stride)

    # 从meter中获得数据
    raw_pseudo_num_per_class = meters[f'raw_pseudo_num_per_class_{mode}'].avg
    select_pseudo_num_per_class = meters[f"select_pseudo_num_per_class_{mode}"].avg
    raw_correct_num_per_class = meters[f"raw_correct_num_per_class_{mode}"].avg
    select_correct_num_per_class = meters[f"select_correct_num_per_class_{mode}"].avg
    raw_acc_per_class = meters[f"raw_acc_per_class_{mode}"].avg
    select_acc_per_class = meters[f"select_acc_per_class_{mode}"].avg
    mask_L_num_per_class = meters[f"mask_L_num_per_class_{mode}"].avg
    mask_U_num_per_class = meters[f"mask_U_num_per_class_{mode}"].avg

    # 1. raw_pseudo_num_per_class
    log_single_stat(writer, raw_pseudo_num_per_class, num_class, f'PD-Num-Raw/{split}-{mode}', stride, log_unit, step)
    # 2. select_pseudo_num_per_class
    log_single_stat(writer, select_pseudo_num_per_class, num_class, f'PD-Num-Sel/{split}-{mode}', stride, log_unit, step)
    # 3. raw_correct_num_per_class
    log_single_stat(writer, raw_correct_num_per_class, num_class, f'Corr-PD-Num-Raw/{split}-{mode}', stride, log_unit, step)
    # 4. select_correct_num_per_class
    log_single_stat(writer, select_correct_num_per_class, num_class, f'Corr-PD-Num-Sel/{split}-{mode}', stride, log_unit, step)
    # 5. raw_acc_per_class
    log_single_stat(writer, raw_acc_per_class, num_class, f'PD-Acc-Raw/{split}-{mode}', stride, log_unit, step)
    # 6. select_acc_per_class
    log_single_stat(writer, select_acc_per_class, num_class, f'PD-Acc-Sel/{split}-{mode}', stride, log_unit, step)
    if split == 'Train':
        # 7. mask_U_num_per_class
        log_single_stat(writer, mask_U_num_per_class, num_class, f'Mask-num-U/{split}-{mode}', stride, log_unit, step)
        # 8. mask_L_num_per_class
        log_single_stat(writer, mask_L_num_per_class, num_class, f'Mask-num-L/{split}-{mode}', stride, log_unit, step)


def log_loss(writer, meters):
    writer.update(loss_total=meters['losses_total'].avg, head='Train-Loss')
    writer.update(loss_L_base=meters['losses_L_base'].avg, head='Train-Loss')
    writer.update(loss_U_base=meters['losses_U_base'].avg, head='Train-Loss')
    writer.update(loss_L_abc=meters['losses_L_abc'].avg, head='Train-Loss')
    writer.update(loss_U_abc=meters['losses_U_abc'].avg, head='Train-Loss')
    writer.update(loss_tcp=meters['losses_tcp'].avg, head='Train-Loss')


def log_single_stat(writer, stat_per_class, num_class, main_tag, stride, log_unit=None, step=-1):
    half_C = int(num_class // 2)
    step = writer.step if step == -1 else step
    log_unit = log_unit if log_unit else get_log_unit(num_class=num_class, stride=stride)
    writer.add_scalars(main_tag=f'{main_tag}-Global', global_step=step,
                       tag_scalar_dict={'all': stat_per_class.sum().item(),
                                        'maj': stat_per_class[:half_C].sum().item(),
                                        'min': stat_per_class[half_C:].sum().item()})

    writer.add_scalars(main_tag=f'{main_tag}-Local', global_step=step,
                       tag_scalar_dict={tag: stat_per_class[idxs].mean().item() for tag, idxs in
                                        log_unit.items()})
