# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import numpy as np
from utils import *
from engine import TrainEngine, ValEngine
from dataset import create_dataset
from model import create_model


def main(args):
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # distributed config
    init_distributed_mode(args)

    # create logger
    if is_main_process():
        log_writer, cfg_logger, train_logger, val_logger, tcp_num_logger, tcp_acc_logger, tcp_get_num_logger = create_logger(args, remove_console=False)
        cfg_logger.info("==> hyperparameters:\n{}".format(args).replace(',', '\n'))
    else:
        log_writer, cfg_logger, train_logger, val_logger, tcp_num_logger, tcp_acc_logger, tcp_get_num_logger = None, None, None, None, None, None, None

    # make output dir
    if not os.path.isdir(args.out) and is_main_process():
        mkdir_p(args.out)

    # set random seed
    set_random_seed(seed=args.seed)

    # Prepare dataset
    labeled_trainloader, unlabeled_trainloader, val_loader, sample_num_per_class, num_class = create_dataset(args, cfg_logger)

    # Model
    model, params = create_model(model_name=args.model, num_class=num_class, pretrained=args.pretrained)
    ema_model, _ = create_model(model_name=args.model, num_class=num_class, ema=True)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    cudnn.benchmark = True
    if cfg_logger is not None:
        cfg_logger.info(f"==> creating {args.model} with abc")
        cfg_logger.info('==> Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    # Loss function and optimizer
    train_criterion = SemiLoss()
    val_criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(args, model_without_ddp)

    lr_scheduler = get_lr_scheduler(args, optimizer)
    ema_optimizer = WeightEMA(model_without_ddp, ema_model, lr=args.lr, alpha=args.ema_decay)

    # Resume from checkpoint
    start_epoch, tcp_state_dict, best_epoch, best_acc = auto_resume(
        args, logger=cfg_logger, model=model_without_ddp, ema_model=ema_model,
        optimizer=optimizer, log_writer=log_writer, scheduler=lr_scheduler)

    # initialize engines
    train_engine = TrainEngine(args=args, labeled_loader=labeled_trainloader, unlabeled_loader=unlabeled_trainloader,
                               model=model,
                               optimizer=optimizer, ema_optimizer=ema_optimizer, semi_loss=train_criterion,
                               sample_num_per_class=sample_num_per_class, num_class=num_class, logger=train_logger,
                               log_writer=log_writer, tcp_state_dict=tcp_state_dict,
                               tcp_num_logger=tcp_num_logger,
                               tcp_get_num_logger=tcp_get_num_logger,
                               tcp_acc_logger=tcp_acc_logger)

    val_model = model if args.disable_ema_model else ema_model
    val_engine = ValEngine(args=args, val_loader=val_loader, model=val_model, criterion=val_criterion,
                           num_class=num_class, logger=val_logger, log_writer=log_writer,
                           train_sample_num_per_class=sample_num_per_class)

    # train and eval
    accs = []
    for _, epoch in enumerate(range(start_epoch, args.epochs)):
        if args.distributed:
            labeled_trainloader.sampler.set_epoch(epoch)
            unlabeled_trainloader.sampler.set_epoch(epoch)

        tcp_state_dict = train_engine.train_one_epoch_fix(epoch=epoch)
        val_stat = val_engine.val_one_epoch_fix(epoch=epoch, best_acc=best_acc, best_epoch=best_epoch, accs=accs)

        if is_main_process():
            best_acc, best_epoch = save_checkpoint(
                cur_acc=val_stat['mean_acc'], best_acc=best_acc, best_epoch=best_epoch,
                state={'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict(),
                       'tcp_state_dict': tcp_state_dict,
                       'ema_state_dict': ema_model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'lr_scheduler': lr_scheduler.state_dict(),
                       'best_epoch': best_epoch, 'best_acc': best_acc, 'tensorboard_step': log_writer.step},
                epoch=epoch + 1, last_epoch=(epoch + 1 == args.epochs),
                save_path=f"{args.out}/ckps", given_save_epoch=args.ckp_given_save_epoch,
                save_freq=args.ckp_save_epoch_freq)

        lr_scheduler.step()
        accs.append(val_stat['mean_acc'])

    try:
        if val_logger is not None:
            val_logger.info("[Mean Val Acc of last 10 Epoch:{mean_acc:.4f}", mean_acc=np.mean(accs[-10:]))
            val_logger.info("[Mean Val Acc of last 20 Epoch:{mean_acc:.4f}", mean_acc=np.mean(accs[-20:]))
    except:
        pass


if __name__ == '__main__':
    my_args = get_args()
    main(my_args)
