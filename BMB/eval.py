# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import numpy as np

from utils import *
from engine import EvalEngine
from dataset import create_dataset
from model import create_model


def main(args):
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # distributed config
    init_distributed_mode(args)

    # create logger
    if is_main_process():
        log_writer, cfg_logger, train_logger, val_logger, _, _, _, _ = create_logger(args, remove_console=False)
        cfg_logger.info("==> hyperparameters:\n{}".format(args).replace(',', '\n'))
    else:
        log_writer, cfg_logger, train_logger, val_logger = None, None, None, None

    # make output dir
    if not os.path.isdir(args.out) and is_main_process():
        mkdir_p(args.out)

    # set random seed
    set_random_seed(seed=args.seed)

    # Prepare dataset
    labeled_trainloader, unlabeled_trainloader, val_loader, sample_num_per_class, num_class = create_dataset(args,
                                                                                                             cfg_logger)

    # Model
    model, params = create_model(model_name=args.model, num_class=num_class, pretrained=args.pretrained)
    # load customize param
    ema_model, _ = create_model(model_name=args.model, num_class=num_class, ema=True)

    # Resume from checkpoint
    checkpoint = torch.load(args.resume, map_location='cpu')
    if args.daso:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = 0
    best_epoch = 0

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    cudnn.benchmark = True
    if cfg_logger is not None:
        cfg_logger.info(f"==> creating {args.model} with abc")
        cfg_logger.info('==> Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    # optimizer
    val_criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model_without_ddp)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    val_model = model if args.disable_ema_model else ema_model
    val_engine = EvalEngine(args=args, val_loader=val_loader, model=val_model, criterion=val_criterion,
                            num_class=num_class, logger=val_logger, log_writer=log_writer,
                            train_sample_num_per_class=sample_num_per_class)
    # train and eval
    # val_stat = val_engine.val_one_epoch_fix(epoch=epoch, best_acc=best_acc, best_epoch=best_epoch)
    val_stat = val_engine.val_one_epoch_fix(epoch=0, best_acc=best_acc, best_epoch=best_epoch)
    if is_main_process():
        print('mean_acc: ', val_stat['mean_acc'])
        print('many_acc: ', val_stat['many_acc'])
        print('medium_acc: ', val_stat['medium_acc'])
        print('few_acc: ', val_stat['few_acc'])


if __name__ == '__main__':
    my_args = get_args()
    assert my_args.pd_distribution_estimate_nepoch * my_args.pd_stat_ema_momentum_wt < 0
    main(my_args)
