import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')

    # model configuration
    parser.add_argument('--model', type=str, default='wideresnet', help='backbone to extract features',
                        choices=['wideresnet', 'resnet', 'resnext', 'resnet_img127', 'resnet_baseline'])
    parser.add_argument('--pretrained', action='store_true', help='load model from  pretrained checkpoints')
    parser.add_argument('--eval_base', action='store_true', help='use base classifier to evaluate not abc classifier')

    # training options
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--val-iteration', type=int, default=500, help='iteration steps per epoch')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='warmup with supervised data only')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--warm_only_base', action='store_true')
    parser.add_argument('--labeled_batch_size', default=64, type=int, metavar='N', help='train/val labeled batch_size')
    parser.add_argument('--unlabeled_batch_size', default=64, type=int, metavar='N', help='train unlabeled batch_size')
    parser.add_argument('--val_batch_size', default=64, type=int, metavar='N', help='val batch_size')
    parser.add_argument('--loss_u_weight', default=1.0, type=float)

    # optimizer
    parser.add_argument('--optim_type', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)

    # learning rate config
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler_type', default='none', type=str, help='learning rate scheduler type',
                        choices=['none', 'step', 'cos'])
    parser.add_argument('--lr_steps', nargs='+', type=int, help='learning steps when use StepLR')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1, help='learning decay rate when use StepLR')

    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='result', type=str, help='Directory to output the result')
    parser.add_argument('--ckp_save_epoch_freq', type=int, default=1000)
    parser.add_argument('--ckp_given_save_epoch', type=int, default=-1)
    parser.add_argument('--writer_log_iter_freq', type=int, default=100)
    parser.add_argument('--train_log_iter_freq', type=int, default=100)
    parser.add_argument('--writer_log_class_stride', type=int, required=True)
    parser.add_argument('--log_backbone', action='store_true')

    # Miscs
    parser.add_argument('--seed', type=int, default=0, help='manual seed')
    parser.add_argument('--many_shot_thr', type=int, default=100, help='many shot threshold in shot accuracy calculation')
    parser.add_argument('--low_shot_thr', type=int, default=20, help='low shot threshold in shot accuracy calculation')

    # Device options
    parser.add_argument('--gpu_ids', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_workers', default=4, type=int, help='number of threads when loading data')

    # Method options
    parser.add_argument('--disable_backbone', action='store_true', help='do not use backbone SSL, such as Fixmatch, MixMatch')
    parser.add_argument('--disable_abc', action='store_true', help='do not use ABC')
    parser.add_argument('--num_max_l', type=int, default=1500, help='Number of samples in the maximal class of labeled data')
    parser.add_argument('--num_max_u', type=int, default=3000, help='Number of samples in the maximal class of unlabeled data')
    parser.add_argument('--imb_ratio_l', type=int, default=100, help='Imbalance ratio for labeled data')
    # parser.add_argument('--imb_ratio_u', type=int, default=-1, help='Imbalance ratio for unlabeled data')
    parser.add_argument('--imb_ratio_u', type=int, default=-1, help='Imbalance ratio for unlabeled data')
    parser.add_argument('--labeled_ratio', type=int, help='labeled ratio for imagenet-lt/imagenet127 dataset')
    parser.add_argument('--step', action='store_true', help='Type of class-imbalance')

    # Hyperparameters for FixMatch
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--disable_ema_model', action='store_true', help='is store_true, we will use model to validate, however, we will enable '
                                                                         'ema_model to use EMA updated model to validate')
    parser.add_argument('--tau', default=0.95, type=float, help='threshold of fixmatch')

    # dataset and imbalanced type
    parser.add_argument('--data_path', type=str, default='./data', help='path to the dataset root dir')
    parser.add_argument('--annotation_file_path', type=str, default='./ann', help='path to the annotation file root dir')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset', choices=['cifar10', 'cifar100', 'imagenet', 'imagenet127', 'small_imagenet127'])
    parser.add_argument('--imb_type', type=str, default='long', help='Long tailed or step imbalanced')

    # improved ABC
    parser.add_argument('--anneal_mask_for_balance', action='store_true', help='whether to anneal mask for balance(unsupervised)')
    parser.add_argument('--anneal_mask_for_balance_L', action='store_true', help='whether to anneal mask for balance(supervised)')
    parser.add_argument('--no_mask_L_for_balance', action='store_true', help='do not mask labeled samples for balance')
    parser.add_argument('--dingding_notifier', action='store_true', help='use dingding to notice best result on phone')
    parser.add_argument('--dingding_threshold', type=float, default=0., help='only send dingding message when surpass this value')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--pd_stat_ema_momentum_wt', type=float, default=-1., help='-1 means no use ema, use every n epoch')
    parser.add_argument('--pd_distribution_estimate_nepoch', type=int, default=-1, help='update pseudo label distribution '
                                                                                        'every n epoch, -1 means use ema')

    # TCP
    parser.add_argument('--tcp_pool_size', type=int, default=0, help='pool size of TCP')
    parser.add_argument('--tcp_get_num', type=int, default=0, help='get n samples from TCP per time')
    parser.add_argument('--tcp_anneal_get_num', action='store_true', help='increase from 0 to get_num')
    parser.add_argument('--tcp_distribution_type', type=str, default='gt', choices=['gt', 'pd_raw', 'pd_select'])
    parser.add_argument('--tcp_loss_weight', type=float, default=1.)
    parser.add_argument('--tcp_sync_input', action='store_true', help='synchronize input sample between GPUs')
    parser.add_argument('--tcp_in_warmup', action='store_true', help='back-propagate tcp loss in warmup epochs')
    parser.add_argument('--tcp_include_strong', action='store_true', help='include strong augmented unlabeled features in TCP pool')
    parser.add_argument('--tcp_include_labeled', action='store_true', help='include labeled features in TCP pool')
    parser.add_argument('--tcp_remove_type', type=str, default='prob', choices=['prob', 'rand', 'fifo', 'inpool'])
    parser.add_argument('--tcp_put_type', type=str, default='prob', choices=['prob', 'rand', 'fifo', 'inpool'])
    parser.add_argument('--tcp_get_type', type=str, default='prob', choices=['prob', 'rand'])
    parser.add_argument('--tcp_balance_power', type=float, default=1)
    parser.add_argument('--tcp_sg', action='store_true')
    parser.add_argument('--tcp_separate_labeled', action='store_true', help='create a separate TCP for labeled data')
    parser.add_argument('--tcp_refresh_after_warm', action='store_true')
    parser.add_argument('--tcp_strong', action='store_true')
    parser.add_argument('--tcp_warmup_epochs', type=int, default=-1)

    # Sample probability distribution
    parser.add_argument('--tcp_sample_fun_type', type=str, default='none', help='sample probability function',
                        choices=['exp_min', 'poly_inv', 'abc'])
    parser.add_argument('--tcp_sample_power', type=float, default=99999)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', help='enable distributed evaluation/validation')

    # adaptive weight
    parser.add_argument('--ada_weight_type', type=str, default='gt',
                        choices=['gt', 'pd_select', 'pd_raw', 'no'],
                        help='distribution information to generate mask for balance for unlabeled data')
    parser.add_argument('--sample_fun_type', type=str, default='exp_min', help='sample probability function',
                        choices=['exp_min', 'poly_inv', 'abc'])
    parser.add_argument('--sample_power', type=float, default=1.)
    parser.add_argument('--ada_weight_anneal_L', action='store_true')
    parser.add_argument('--ada_weight_anneal_U', action='store_true')

    parser.add_argument('--crop_size', default=112, type=int)

    # parser.add_argument('--aux_stop_gradient', action='store_true', help='if True, aux. classifier will stop_gradient to the encoder, including'
    #                                                                      'loss_tcp and loss_abc_L, loss_abc_U')
    parser.add_argument('--base_pd_from_aux', action='store_true', help='if True, auxiliary classifier will generate pseudo-labels for '
                                                                        'the base classifier')

    parser.add_argument('--max_num_l', type=int, default=600)
    parser.add_argument('--max_num_u', type=int, default=300)

    parser.add_argument('--tcp_ablate_feat', action='store_true')
    parser.add_argument('--tcp_use_weak', action='store_true')
    parser.add_argument('--tcp_use_strong', action='store_true')
    parser.add_argument('--tcp_use_weak_strong', action='store_true')
    parser.add_argument('--tcp_cross_pd', action='store_true')

    parser.add_argument('--tcp_store_img', action='store_true')

    args = parser.parse_args()
    # 进行一些默认设置，以保证与之前的版本兼容
    if args.tcp_sample_fun_type == 'none':
        args.tcp_sample_fun_type = args.sample_fun_type
    if args.tcp_sample_power == 99999:
        args.tcp_sample_power = args.sample_power

    # 校验：至少选择一种分布估计方式
    assert args.pd_distribution_estimate_nepoch * args.pd_stat_ema_momentum_wt < 0

    if args.tcp_warmup_epochs == -1:
        args.tcp_warmup_epochs = args.warmup_epochs

    return args
