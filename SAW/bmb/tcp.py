import torch
from bmb import label2onehot, get_index, generate_probs_per_class, log_single_stat, dist


class TailClassPool(object):
    def __init__(self, pool_size, balance_power, sample_power, class_distribution):
        self.pool_size = pool_size
        self.balance_power = balance_power
        self.sample_power = sample_power
        self.sample_num = 0  # in-pool sample num
        self.feat_dim = 128
        self.class_num = len(class_distribution)
        self.class_distribution = class_distribution
        self.feature_pool = torch.zeros(self.pool_size, self.feat_dim)
        self.label_pool = torch.ones(self.pool_size) * -1

    @torch.no_grad()
    def put_samples(self, input_features, input_labels):
        """
        Args:
            input_features: [B, feat_dim] features of samples
            input_labels: [B] labels of samples
        """
        input_labels = input_labels.detach()

        B = input_features.shape[0]
        input_onehot = label2onehot(label=input_labels, batch=B, num_class=self.class_num, to_cuda=True)

        # 按照类别样本数量分布，选取本次需要加入的样本
        inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=self.class_num)
        in_pool_num_per_class = torch.sum(inpool_onehot, dim=0)
        put_probs_per_class = generate_probs_per_class(sample_fun_type='poly_inv',
                                                       stat_per_class=in_pool_num_per_class,
                                                       power=self.balance_power,
                                                       norm_type='local', )

        put_probs_per_sample = input_onehot @ put_probs_per_class  # [B,C]x[C]->[B]

        put_mask = torch.bernoulli(put_probs_per_sample)  # [B], 1 means put this sample into TCP
        put_idx = torch.where(put_mask == 1)  # index of selected samples

        put_features = input_features[put_idx].cuda()
        put_labels = input_labels[put_idx].cuda()

        # 判断是否还有空位，若位置不够，则选择并移除一些样本，然后把新样本加入到pool中
        put_num = put_features.shape[0]  # 本次选中的需要加入的样本数
        assert put_num <= self.pool_size, "Too many samples selected"
        empty_num = self.pool_size - self.sample_num  # 剩余的空位
        if empty_num >= put_num:
            # 空位足够，直接加入
            self.feature_pool[self.sample_num:self.sample_num + put_num, :] = put_features
            self.label_pool[self.sample_num:self.sample_num + put_num] = put_labels
            self.sample_num += put_num
            remove_num = 0
            remove_labels = None
        else:
            # 空位不够了，需要移除一些样本
            remove_num = put_num - empty_num  # 需要移除的样本数
            C = self.class_num
            inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=C)

            remove_probs_per_sample = 1 - inpool_onehot @ put_probs_per_class
            remove_idx = get_index(probs=remove_probs_per_sample, get_num=remove_num)
            remove_labels = self.label_pool[remove_idx]

            # 将新样本加入到pool中并移除被选中的旧样本
            empty_idx = torch.tensor([]) if empty_num == 0 else torch.arange(self.pool_size)[
                                                                self.sample_num:]  # 空闲位置的index
            insert_idx = torch.cat([empty_idx.cuda(), remove_idx.cuda()])  # 新样本插入的位置index
            self.feature_pool[insert_idx.long()] = put_features.cpu()
            self.label_pool[insert_idx.long()] = put_labels.float().cpu()
            # self.gt_pool[insert_idx.long()] = put_gt.float().cpu()
            self.sample_num = self.sample_num - remove_num + put_num
        return remove_num, remove_labels

    @torch.no_grad()
    def get_samples(self, get_num):
        """
        Args:
            get_num: 需要从pool中采样的样本数量
            class_distribution: 当前的各类别样本分布, [C]
        """
        get_num = self.sample_num if get_num > self.sample_num else get_num
        C = self.class_num
        inpool_onehot = label2onehot(label=self.label_pool[:self.sample_num], batch=self.sample_num, num_class=C)

        # 按照类别样本数量分布，选取本次需要加入的样本
        get_probs_per_class = generate_probs_per_class(sample_fun_type='poly_inv',
                                                       stat_per_class=self.class_distribution,
                                                       power=self.sample_power,
                                                       norm_type='global', )
        get_probs_per_sample = inpool_onehot @ get_probs_per_class  # [sample_num,C]x[C]->[sample_num]
        get_idx = get_index(probs=get_probs_per_sample, get_num=get_num)  # 选中的样本的index
        get_features = self.feature_pool[get_idx]
        get_labels = self.label_pool[get_idx]

        return get_features.cuda(), get_labels.cuda(), get_num
