import torch

import numpy as np
import utils.dist as dist
from torch.utils.data import DataLoader
from .utils import make_imbalanced_data


def create_dataset(args, logger=None):
    """
    create the imbalanced dataset
    Returns:
        labeled_trainloader:
        unlabeled_trainloader:
        val_loader:
        sample_num_per_class: number of samples in each class of labeled and unlabeled dataset
                            {'labeled': ., 'unlabeled':.}
        num_class: how many classes are in this dataset
    """

    if args.dataset == 'cifar10':
        from .fix_cifar10 import get_cifar10 as get_dataset
        num_class = 10
    elif args.dataset == 'svhn':
        from .fix_svhn import get_svhn as get_dataset
        num_class = 10
    elif args.dataset == 'cifar100':
        from .fix_cifar100 import get_cifar100 as get_dataset
        num_class = 100
    elif args.dataset == 'imagenet':
        from .imagenet import get_imagenet as get_dataset
        num_class = 1000
    elif args.dataset == 'imagenet127':
        from .imagenet127 import get_imagenet as get_dataset
        num_class = 127
    elif args.dataset == 'small_imagenet127':
        from .small_imagenet127 import get_small_imagenet as get_dataset
        num_class = 127
    else:
        raise NotImplementedError('This dataset is not implemented yet')

    if args.dataset == 'imagenet':
        train_labeled_set, train_unlabeled_set, val_set, sample_num_per_class = get_dataset(
            root=args.data_path,
            annotation_file_train_labeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_labeled.txt',
            annotation_file_train_unlabeled=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_unlabeled.txt',
            annotation_file_val=f'{args.annotation_file_path}/ImageNet_LT_val.txt',
            num_per_class=f'{args.annotation_file_path}/ImageNet_LT_train_semi_{int(args.labeled_ratio)}_sample_num.txt')
    elif args.dataset == 'imagenet127':
        train_labeled_set, train_unlabeled_set, val_set, sample_num_per_class = get_dataset(
            root=args.data_path,
            annotation_file_train_labeled=f'{args.annotation_file_path}/ImageNet127_LT_train_semi_{int(args.labeled_ratio)}_labeled.txt',
            annotation_file_train_unlabeled=f'{args.annotation_file_path}/ImageNet127_LT_train_semi_{int(args.labeled_ratio)}_unlabeled.txt',
            annotation_file_val=f'{args.annotation_file_path}/ImageNet127_LT_val.txt',
            num_per_class=f'{args.annotation_file_path}/ImageNet127_LT_train_semi_{int(args.labeled_ratio)}_sample_num.txt',
            crop_size=args.crop_size
        )
    elif args.dataset == 'small_imagenet127':
        small_127_root = f'{args.data_path}/res{args.crop_size}'
        train_labeled_set, train_unlabeled_set, val_set, sample_num_per_class = get_dataset(
            root=small_127_root,
            img_size=args.crop_size,
            labeled_percent=args.labeled_ratio,
            seed=args.seed)

    else:
        labeled_sample_num_per_class = make_imbalanced_data(
            args.num_max_l,
            num_class,
            args.imb_ratio_l,
            args.imb_type)
        unlabeled_sample_num_per_class = make_imbalanced_data(
            args.num_max_u,
            num_class,
            args.imb_ratio_u,
            args.imb_type)

        sample_num_per_class = {
            'labeled': labeled_sample_num_per_class,
            'unlabeled': unlabeled_sample_num_per_class
        }

        train_labeled_set, train_unlabeled_set, val_set = get_dataset(args.data_path,
                                                                      labeled_sample_num_per_class,
                                                                      unlabeled_sample_num_per_class,
                                                                      args=args)

    if logger is not None:
        logger.info('==> number of labeled samples in each class:\n{}', np.array(sample_num_per_class['labeled']))
        logger.info('==> number of unlabeled samples in each class:\n{}', np.array(sample_num_per_class['unlabeled']))
        logger.info("Total Train Labeled Sample:{} / Total Train UnLabeled Sample:{}",
                    len(train_labeled_set),
                    len(train_unlabeled_set))

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        labeled_train_sampler = torch.utils.data.DistributedSampler(
            dataset=train_labeled_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        unlabeled_train_sampler = torch.utils.data.DistributedSampler(
            dataset=train_unlabeled_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(val_set) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            val_sampler = torch.utils.data.DistributedSampler(
                dataset=val_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            val_sampler = torch.utils.data.SequentialSampler(val_set)
        if logger is not None:
            logger.info("Sampler_train_labeled = %s" % str(labeled_train_sampler))
            logger.info("Sampler_train_unlabeled = %s" % str(unlabeled_train_sampler))
            logger.info("Sampler_val = %s" % str(val_sampler))
    else:
        labeled_train_sampler = torch.utils.data.RandomSampler(train_labeled_set)
        unlabeled_train_sampler = torch.utils.data.RandomSampler(train_unlabeled_set)
        val_sampler = torch.utils.data.SequentialSampler(val_set)
        if logger is not None:
            logger.info("You are not used distributed training!")

    labeled_trainloader = DataLoader(train_labeled_set,
                                     batch_size=args.labeled_batch_size,
                                     sampler=labeled_train_sampler,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_set,
                                       batch_size=args.unlabeled_batch_size,
                                       sampler=unlabeled_train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)

    return labeled_trainloader, unlabeled_trainloader, val_loader, sample_num_per_class, num_class
