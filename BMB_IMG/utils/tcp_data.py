import torch
import numpy as np
from utils import label2onehot, get_index, generate_probs_per_class, log_single_stat, dist


class TailClassPool_Data(object):
    def __init__(self, log_writer, args, mode, num_logger, acc_logger, get_logger):
        self.pool_size = args.tcp_pool_size
        self.sample_num = 0  # in-pool sample num
        self.num_logger = num_logger
        self.acc_logger = acc_logger
        self.get_logger = get_logger
        if args.model == 'wideresnet':
            self.feat_dim = 128
        elif args.model in ['resnet', 'resnet_img127']:
            self.feat_dim = 2048
        else:
            raise KeyError
        self.feature_pool = torch.zeros(self.pool_size, self.feat_dim)
        self.label_pool = torch.ones(self.pool_size) * -1
        self.gt_pool = torch.ones(self.pool_size) * -2
        self.log_writer = log_writer
        self.args = args
        self.mode = mode

    @torch.no_grad()
    def put_samples(self, class_distribution, input_features, input_labels, gt_labels):
        """
        Args:
            class_distribution: [C] number of samples per class
            input_features: [B, feat_dim] features of samples
            input_labels: [B] labels of samples
            gt_labels
        """
        input_labels = input_labels.detach()

        B, C = input_features.shape[0], class_distribution.shape[0]
        input_onehot = label2onehot(label=input_labels, batch=B, num_class=C, to_cuda=True)

        # 按照类别样本数量分布，选取本次需要加入的样本
        if self.args.tcp_put_type == 'inpool':
            inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=C)
            in_pool_num_per_class = torch.sum(inpool_onehot, dim=0)
            put_probs_per_class = generate_probs_per_class(sample_fun_type=self.args.tcp_sample_fun_type,
                                                           stat_per_class=in_pool_num_per_class,
                                                           power=self.args.tcp_balance_power,
                                                           norm_type='local', )
        elif self.args.tcp_put_type == 'prob':
            # put_probs_per_class = generate_probs_per_class(sample_fun_type=self.args.tcp_sample_fun_type,
            #                                                stat_per_class=class_distribution,
            #                                                power=self.args.tcp_sample_power,
            #                                                norm_type='local', )

            raise NotImplementedError
        else:
            raise NotImplementedError

        put_probs_per_sample = input_onehot @ put_probs_per_class  # [B,C]x[C]->[B]

        put_mask = torch.bernoulli(put_probs_per_sample)  # [B], 1 means put this sample into TCP
        put_idx = torch.where(put_mask == 1)  # index of selected samples

        put_features = input_features[put_idx].cuda()
        put_labels = input_labels[put_idx].cuda()
        put_gt = gt_labels[put_idx].cuda()

        # 判断是否还有空位，若位置不够，则选择并移除一些样本，然后把新样本加入到pool中
        put_num = put_features.shape[0]  # 本次选中的需要加入的样本数
        assert put_num <= self.pool_size, "Too many samples selected"
        empty_num = self.pool_size - self.sample_num  # 剩余的空位
        if empty_num >= put_num:
            # 空位足够，直接加入
            self.feature_pool[self.sample_num:self.sample_num + put_num, :] = put_features
            self.label_pool[self.sample_num:self.sample_num + put_num] = put_labels
            self.gt_pool[self.sample_num:self.sample_num + put_num] = put_gt
            self.sample_num += put_num
            remove_num = 0
            remove_labels = None
        else:
            # 空位不够了，需要移除一些样本
            remove_num = put_num - empty_num  # 需要移除的样本数
            inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=C)

            if self.args.tcp_remove_type == 'prob':
                raise NotImplementedError
            elif self.args.tcp_remove_type == 'rand':
                raise NotImplementedError
            elif self.args.tcp_remove_type == 'inpool':
                remove_probs_per_sample = 1 - inpool_onehot @ put_probs_per_class
            else:
                raise NotImplementedError
            remove_idx = get_index(probs=remove_probs_per_sample, get_num=remove_num)
            remove_labels = self.label_pool[remove_idx]

            # 将新样本加入到pool中并移除被选中的旧样本
            empty_idx = torch.tensor([]) if empty_num == 0 else torch.arange(self.pool_size)[
                                                                self.sample_num:]  # 空闲位置的index
            insert_idx = torch.cat([empty_idx.cuda(), remove_idx.cuda()])  # 新样本插入的位置index
            self.feature_pool[insert_idx.long()] = put_features.cpu()
            self.label_pool[insert_idx.long()] = put_labels.float().cpu()
            self.gt_pool[insert_idx.long()] = put_gt.float().cpu()
            self.sample_num = self.sample_num - remove_num + put_num
        return remove_num, remove_labels

    @torch.no_grad()
    def get_samples(self, get_num, class_distribution):
        """
        Args:
            get_num: 需要从pool中采样的样本数量
            class_distribution: 当前的各类别样本分布, [C]
        """
        get_num = self.sample_num if get_num > self.sample_num else get_num
        C = class_distribution.shape[0]
        inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=C)

        # 按照类别样本数量分布，选取本次需要加入的样本
        if self.args.tcp_get_type == 'prob':
            get_probs_per_class = generate_probs_per_class(sample_fun_type=self.args.tcp_sample_fun_type,
                                                           stat_per_class=class_distribution,
                                                           power=self.args.tcp_sample_power,
                                                           norm_type='global', )
        elif self.args.tcp_get_type == 'rand':
            get_probs_per_class = torch.ones(self.sample_num)
        else:
            raise NotImplementedError
        get_probs_per_sample = inpool_onehot @ get_probs_per_class  # [sample_num,C]x[C]->[sample_num]
        get_idx = get_index(probs=get_probs_per_sample, get_num=get_num)  # 选中的样本的index
        get_features = self.feature_pool[get_idx]
        get_labels = self.label_pool[get_idx]

        return get_features.cuda(), get_labels.cuda(), get_num

    @torch.no_grad()
    def all_gather_obj_put_sample(self, feats, labels):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.args.distributed:
            obj = {'feats': feats, 'labels': labels}
            obj_list = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(obj_list, obj)

            feats_list = [obj['feats'].cuda() for obj in obj_list]
            labels_list = [obj['labels'].cuda() for obj in obj_list]

            feats_gathered = torch.cat(feats_list, dim=0)
            labels_gathered = torch.cat(labels_list, dim=0)
        else:
            return feats, labels

        return feats_gathered, labels_gathered

    def log_inpool_stat(self, num_class, epoch):
        inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num,
                                     num_class=num_class)
        in_pool_num_per_class = inpool_onehot.sum(dim=0)
        log_single_stat(self.log_writer, in_pool_num_per_class, num_class, f'{self.mode}-TCP/Inpool-Num',
                        stride=self.args.writer_log_class_stride)
        #  log in-pool-num to tcp_num_logger
        # 将 Tensor 转换为 NumPy 数组
        array = in_pool_num_per_class.cpu().numpy()
        # 将数组转换为字符串，并记录在日志中
        formatter = {"float_kind": lambda x: "{:.4f}".format(x)}
        array_string = np.array2string(array, separator=",", formatter=formatter, max_line_width=np.inf)
        # 去掉方括号
        array_string = array_string.strip('[]')
        self.num_logger.info(array_string)

        # log in-pool-acc to tcp_acc_logger
        if self.sample_num > 0:
            acc = (self.gt_pool == self.label_pool).sum() / self.sample_num
            acc *= 100
        else:
            acc = 0.
        self.acc_logger.info('Epoch:{epoch:03d}-Acc:{acc:.3f}', epoch=epoch+1, acc=acc)

    def log_get_stat(self, num_class, get_labels, get_num):
        get_labels_onehot = label2onehot(label=get_labels, batch=get_num, num_class=num_class)
        get_num_per_class = get_labels_onehot.sum(dim=0)
        log_single_stat(self.log_writer, get_num_per_class, num_class, f'{self.mode}-TCP/Get-Num',
                        stride=self.args.writer_log_class_stride)

        #  log in-pool-num to tcp_num_logger
        # 将 Tensor 转换为 NumPy 数组
        array = get_num_per_class.cpu().numpy()
        # 将数组转换为字符串，并记录在日志中
        formatter = {"float_kind": lambda x: "{:.4f}".format(x)}
        array_string = np.array2string(array, separator=",", formatter=formatter, max_line_width=np.inf)
        # 去掉方括号
        array_string = array_string.strip('[]')
        self.get_logger.info(array_string)

    def log_remove_stat(self, num_class, remove_labels, remove_num):
        remove_labels_onehot = label2onehot(label=remove_labels, batch=remove_num, num_class=num_class)
        remove_num_per_class = remove_labels_onehot.sum(dim=0)
        log_single_stat(self.log_writer, remove_num_per_class, num_class, f'{self.mode}-TCP/Remove-Num',
                        stride=self.args.writer_log_class_stride)

    @property
    def state_dict(self):
        """get state dict for checkpoint"""
        state_dict = {
            'pool_size': self.pool_size,
            'sample_num': self.sample_num,
            'feature_pool': self.feature_pool,
            'label_pool': self.label_pool
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """load state from checkpoint"""
        self.pool_size = state_dict['pool_size']
        self.feature_pool = state_dict['feature_pool']
        self.label_pool = state_dict['label_pool']
        self.sample_num = state_dict['sample_num']
