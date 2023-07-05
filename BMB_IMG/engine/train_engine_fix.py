import sys
import time

import torch
import torch.nn.functional as F

sys.path.append('../')
from utils import *
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

torch.autograd.set_detect_anomaly(True)


class TrainEngine(object):
    def __init__(self, args, labeled_loader=None, unlabeled_loader=None, model=None, tcp_resume=False,
                 optimizer=None, ema_optimizer=None, semi_loss: SemiLoss = None, tcp_state_dict=None,
                 sample_num_per_class=None, num_class=None, logger=None, log_writer=None, tcp_num_logger=None,
                 tcp_acc_logger=None, tcp_get_num_logger=None, mis_logger=None):
        self.args = args
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)
        self.model = model
        self.optimizer = optimizer
        self.ema_optimizer = ema_optimizer
        self.semi_loss = semi_loss
        self.sample_num_per_class = sample_num_per_class
        self.num_class = num_class
        self.logger = logger
        self.log_writer = log_writer
        self.tcp_num_logger = tcp_num_logger
        self.tcp_acc_logger = tcp_acc_logger
        self.tcp_get_num_logger = tcp_get_num_logger
        self.pseudo_distribution_meters = init_pd_distribution_meters(dim=num_class)
        if self.args.tcp_store_img:
            self.tcp = TailClassPool_Image(args=args, log_writer=log_writer, mode='U')
        else:
            raise KeyError
            # self.tcp = TailClassPool(args=args, log_writer=log_writer, mode='U', num_logger=self.tcp_num_logger,
            #                          acc_logger=self.tcp_acc_logger, get_logger=self.tcp_get_num_logger)
        if tcp_state_dict and tcp_resume:
            raise KeyError
            # self.tcp.load_state_dict(state_dict=tcp_state_dict)

        self.mis_logger = mis_logger

    def train_one_epoch_fix(self, epoch):

        # 如果没使用EMA而是使用every n epoch进行分布估计
        if self.args.pd_distribution_estimate_nepoch > 0 and epoch % self.args.pd_distribution_estimate_nepoch == 0:
            self.pseudo_distribution_meters = init_pd_distribution_meters(dim=self.num_class)
        if epoch == self.args.warmup_epochs:
            self.pseudo_distribution_meters = init_pd_distribution_meters(dim=self.num_class)
            if self.args.tcp_refresh_after_warm:
                if self.args.tcp_store_img:
                    self.tcp = TailClassPool_Image(args=self.args, log_writer=self.log_writer, mode='U')
                else:
                    self.tcp = TailClassPool(args=self.args, log_writer=self.log_writer, mode='U', num_logger=self.tcp_num_logger, acc_logger=self.tcp_acc_logger, get_logger=self.tcp_get_num_logger)

        per_epoch_meters = init_average_meters(dim=self.num_class)

        # begin training loop
        self.model.train()
        loop = tqdm(range(self.args.val_iteration), colour='blue', leave=False)
        for batch_idx, _ in enumerate(loop):
            end = time.time()

            # get data
            batch_data = self.get_batch_data(to_cuda=True)
            per_epoch_meters['data_time'].update(time.time() - end)

            imgs_L, targets_L, indice_L = batch_data['labeled']
            imgs_U_weak, imgs_U_strong1, imgs_U_strong2, targets_U, indice_U = batch_data['unlabeled']

            batch_size_L, batch_size_U = imgs_L.size(0), imgs_U_weak.size(0)

            # forward_pass
            imgs_all = torch.cat([imgs_L, imgs_U_weak, imgs_U_strong1, imgs_U_strong2])
            feats_all, logits_abc_all, logits_base_all = self.model(imgs_all)
            feats_L = feats_all[:batch_size_L]
            feats_U_weak = feats_all[batch_size_L:batch_size_L + batch_size_U]
            feats_U_strong1 = feats_all[batch_size_L + batch_size_U:batch_size_L + batch_size_U * 2]
            feats_U_strong2 = feats_all[batch_size_L + batch_size_U * 2:]

            # Base algorithm: FixMatch
            if self.args.disable_backbone:
                loss_L_base, loss_U_base = torch.zeros(1).cuda().detach(), torch.zeros(1).cuda().detach()
                pseudo_stat_base = None
            else:
                loss_L_base, loss_U_base, soft_pseudo_base, mask_U_for_balance_base, mask_L_for_balance_base = self.compute_loss(
                    meters=self.pseudo_distribution_meters, logits_all=logits_base_all, batch_size_L=batch_size_L,
                    batch_size_U=batch_size_U, targets_L=targets_L, mode='base', epoch=epoch, logits_abc_all=logits_abc_all)
                #  计算pseudo-label的统计信息
                pseudo_stat_base = analysis_train_pseudo_labels(true_labels_U=targets_U, soft_pseudo_U=soft_pseudo_base,
                                                                true_labels_L=targets_L,
                                                                threshold=self.args.tau, args=self.args,
                                                                mask_L_for_balance=mask_L_for_balance_base,
                                                                mask_U_for_balance=mask_U_for_balance_base)

            # Our algorithm: ABC
            if self.args.disable_abc:
                loss_L_abc, loss_U_abc = torch.zeros(1).cuda().detach(), torch.zeros(1).cuda().detach()
                pseudo_stat_abc = None
            else:
                loss_L_abc, loss_U_abc, soft_pseudo_abc, mask_U_for_balance_abc, mask_L_for_balance_abc = self.compute_loss(
                    meters=self.pseudo_distribution_meters, logits_all=logits_abc_all, batch_size_L=batch_size_L,
                    batch_size_U=batch_size_U, targets_L=targets_L, mode='abc', epoch=epoch)
                #  计算pseudo-label的统计信息
                pseudo_stat_abc = analysis_train_pseudo_labels(true_labels_U=targets_U, soft_pseudo_U=soft_pseudo_abc,
                                                               true_labels_L=targets_L,
                                                               threshold=self.args.tau, args=self.args,
                                                               mask_L_for_balance=mask_L_for_balance_abc,
                                                               mask_U_for_balance=mask_U_for_balance_abc)

            # Tail Class Pool
            if self.args.tcp_pool_size == 0:
                loss_tcp = torch.zeros(1).cuda().detach()
                loss_tcp_labeled = torch.zeros(1).cuda().detach()
            else:
                logits_strong1_abc = logits_abc_all[batch_size_L + batch_size_U:batch_size_L + batch_size_U * 2]
                logits_strong2_abc = logits_abc_all[batch_size_L + batch_size_U * 2:]
                with torch.no_grad():
                    soft_pseudo_strong1 = torch.softmax(logits_strong1_abc, dim=1).detach()
                    soft_pseudo_strong2 = torch.softmax(logits_strong2_abc, dim=1).detach()

                if self.args.tcp_strong:
                    # 使用strong伪标签 (Default)
                    tcp_feats = torch.cat([feats_U_strong1, feats_U_strong2], dim=0)
                    tcp_labels = torch.cat([soft_pseudo_strong1, soft_pseudo_strong2], dim=0)
                    tcp_gt = torch.cat([targets_U, targets_U], dim=0)
                    tcp_indice = torch.cat([indice_U, indice_U], dim=0)
                else:
                    tcp_feats = feats_U_weak
                    tcp_labels = soft_pseudo_abc
                    tcp_gt = targets_U
                    tcp_indice = indice_U

                loss_tcp, tcp_get_num = self.process_tcp(soft_pseudo=tcp_labels, input_features=tcp_feats, epoch=epoch,
                                            input_gt=tcp_gt, indice=tcp_indice)

                loss_tcp_labeled = torch.zeros(1).cuda().detach()

            #  计算损失值
            loss_total = loss_L_base + loss_L_abc
            if epoch >= self.args.warmup_epochs:
                loss_total = loss_total + (loss_U_base + loss_U_abc) * self.args.loss_u_weight
            if epoch >= self.args.tcp_warmup_epochs:
                loss_total = loss_total + (loss_tcp + loss_tcp_labeled) * self.args.tcp_loss_weight

            # update meters
            self.pseudo_distribution_meters = update_pseudo_distribution_meters(
                args=self.args, ema_momentum_wt=self.args.pd_stat_ema_momentum_wt,
                meters=self.pseudo_distribution_meters,
                stats_abc=pseudo_stat_abc, stats_base=pseudo_stat_base)
            per_epoch_meters = update_loss_average_meters(meters=per_epoch_meters, losses={
                'losses_total': loss_total.item(), 'losses_L_base': loss_L_base.item(),
                'losses_U_base': loss_U_base.item(), 'losses_L_abc': loss_L_abc.item(),
                'losses_U_abc': loss_U_abc.item(), 'losses_tcp': loss_tcp.item()})
            per_epoch_meters = update_average_meters(meters=per_epoch_meters, stats_abc=pseudo_stat_abc,
                                                     stats_base=pseudo_stat_base)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            if not self.args.disable_ema_model:
                self.ema_optimizer.step()

            per_epoch_meters['batch_time'].update(time.time() - end - per_epoch_meters['data_time'].val)

            iter_time = time.time() - end

            if self.mis_logger:
                self.mis_logger.info(
                    "{time_per_iter:.3f}, {get_num_per_iter:>3d}",
                    time_per_iter=iter_time,
                    get_num_per_iter=tcp_get_num,
                )

            # log loss and pseudo statistics to tensorboard
            if self.log_writer:
                if (self.log_writer.step + 1) % self.args.writer_log_iter_freq == 0:
                    writer_start_time = time.time()
                    log_loss(writer=self.log_writer, meters=per_epoch_meters)
                    if not self.args.disable_backbone and self.args.log_backbone:
                        log_stats(writer=self.log_writer, meters=per_epoch_meters, mode='base',
                                  num_class=self.num_class, split='Train', stride=self.args.writer_log_class_stride)
                    if not self.args.disable_abc:
                        log_stats(writer=self.log_writer, meters=per_epoch_meters, mode='abc', num_class=self.num_class,
                                  split='Train', stride=self.args.writer_log_class_stride)
                    per_epoch_meters['writer_time'].update(time.time() - writer_start_time)
                self.log_writer.set_step()

            if self.logger and ((batch_idx + 1) % self.args.train_log_iter_freq == 0):
                self.logger.info(
                    "[Train@{epoch:03d}-{iter:03d}/{iters:03d}] [lr:{lr:.4f}] [D:{data:.3f}/B:{batch:.3f}/W:{writer:.3f}]"
                    " [all:{total:.4f}/L_base:{L_base:.4f}/U_base:{U_base:.4f}/L_abc:{L_abc:.4f}/U_abc:{U_abc:.4f}/tcp:{tcp:.4f}]",
                    epoch=epoch + 1, iter=batch_idx + 1, iters=self.args.val_iteration,
                    lr=self.optimizer.param_groups[0]['lr'],
                    writer=per_epoch_meters['writer_time'].sum.item(),
                    data=per_epoch_meters['data_time'].sum.item(),
                    batch=per_epoch_meters['batch_time'].sum.item(),
                    total=loss_total.item(), L_base=loss_L_base.item(), U_base=loss_U_base.item(),
                    L_abc=loss_L_abc.item(), U_abc=loss_U_abc.item(), tcp=loss_tcp.item())

        return self.tcp.state_dict

    def process_tcp(self, soft_pseudo, input_features, epoch, input_gt, indice):
        # 使用TCP
        if self.args.tcp_distribution_type == 'pd_raw':
            num_per_class = self.pseudo_distribution_meters['raw_pseudo_num_per_class_abc']
        elif self.args.tcp_distribution_type == 'pd_select':
            num_per_class = self.pseudo_distribution_meters['select_pseudo_num_per_class_abc']
        elif self.args.tcp_distribution_type == 'gt':
            num_per_class = self.sample_num_per_class['labeled']
        else:
            raise KeyError

        # use every n epoch estimate
        if self.args.pd_distribution_estimate_nepoch > 0 and self.args.tcp_distribution_type != 'gt':
            distribution = num_per_class.avg
        # use ema estimate
        elif self.args.pd_stat_ema_momentum_wt > 0 and self.args.tcp_distribution_type != 'gt':
            distribution = num_per_class.val
        elif self.args.tcp_distribution_type == 'gt':
            distribution = torch.tensor(num_per_class, dtype=torch.float64)
        else:
            raise KeyError

        # 进程间进行同步，以保证input feature的多样性
        # FIXME: 但是由于不同卡上进行不同的随机采用过程，实际上input，remove和get还是不同的
        if self.args.tcp_sync_input:
            soft_pseudo_gathered = concat_all_gather(soft_pseudo)
            input_features_gathered = concat_all_gather(input_features)
            indice_gathered = concat_all_gather(indice)
            input_gt_gathered = concat_all_gather(indice)
        else:
            soft_pseudo_gathered, input_features_gathered, indice_gathered, input_gt_gathered = soft_pseudo, input_features, indice, input_gt

        max_probs, hard_labels = torch.max(soft_pseudo_gathered, dim=1)

        pass_threshold = max_probs.ge(self.args.tau)  # one means pass
        input_idx = torch.where(pass_threshold == 1)  # 超过threshold的样本的坐标
        input_feat = input_features_gathered[input_idx]
        input_labels = hard_labels[input_idx]
        input_gt = input_gt_gathered[input_idx]
        input_indice = indice_gathered[input_idx]

        remove_num, remove_labels = self.tcp.put_samples(class_distribution=distribution, input_features=input_feat,
                                                         input_labels=input_labels, gt_labels=input_gt,
                                                         input_indice=input_indice)

        # anneal get-num
        if self.args.tcp_anneal_get_num:
            get_num = int(self.args.tcp_get_num * (epoch / self.args.epochs))
        else:
            get_num = self.args.tcp_get_num

        tcp_get_indice, tcp_get_labels, tcp_get_gts, tcp_get_num = self.tcp.get_samples(get_num=get_num, class_distribution=distribution)

        # log TCP data into Tensorboard
        if self.log_writer and (self.log_writer.step % self.args.writer_log_iter_freq == 0):
            self.tcp.log_inpool_stat(num_class=self.num_class, epoch=epoch)
            self.tcp.log_get_stat(num_class=self.num_class, get_labels=tcp_get_labels, get_num=tcp_get_num)
            if remove_num > 0:
                self.tcp.log_remove_stat(num_class=self.num_class, remove_labels=remove_labels, remove_num=remove_num)

        # calculate tcp loss
        if tcp_get_num > 0:
            # 根据indice加载图片和标签
            tcp_set = Subset(self.unlabeled_loader.dataset, tcp_get_indice.int().tolist())
            tcp_loader = DataLoader(tcp_set, batch_size=len(tcp_set), shuffle=False)

            if self.args.tcp_store_img_strong:
                (_, tcp_imgs, _), gt_labels, _ = iter(tcp_loader).next()
            else:
                (tcp_imgs, _, _), gt_labels, _ = iter(tcp_loader).next()

            # if dist.get_rank() == 0:
            #     print(f'Get Num: {tcp_get_num}')
            #     print(f'True gt: {gt_labels}')
            #     print(f'TCP gt: {tcp_get_gts}')
            # assert tcp_get_gts == gt_labels

            tcp_imgs, tcp_get_labels = tcp_imgs.cuda(), tcp_get_labels.cuda()

            tcp_feats = self.model.module.extract_feature(tcp_imgs)
            if self.args.tcp_store_img_detach:
                tcp_feats = tcp_feats.detach()
            tcp_logits_abc = self.model.module.fc_abc(tcp_feats)

            tcp_get_labels = label2onehot(tcp_get_labels, tcp_get_num, self.num_class)
            loss_tcp = -torch.mean(torch.sum(F.log_softmax(tcp_logits_abc, dim=1) * tcp_get_labels, dim=1))

        else:
            loss_tcp = torch.zeros(1).cuda()

        return loss_tcp, tcp_get_num

    def compute_loss(self, meters, logits_all, batch_size_L, batch_size_U, targets_L, mode, epoch, logits_abc_all=None):
        """targets_L 形如[B]，还没有转化为one-hot向量"""
        logits_L = logits_all[:batch_size_L]
        logits_U_weak = logits_all[batch_size_L:batch_size_L + batch_size_U]
        logits_U_strong1 = logits_all[batch_size_L + batch_size_U:batch_size_L + batch_size_U * 2]
        logits_U_strong2 = logits_all[batch_size_L + batch_size_U * 2:]

        # generate soft_pseudo_label
        with torch.no_grad():
            soft_pseudo_targets = torch.softmax(logits_U_weak, dim=1).detach()

        # generate hard_pseudo_label
        max_probs, hard_pseudo_targets = torch.max(soft_pseudo_targets, dim=1)

        # convert to one-hot, [B]->[B,C]
        targets_L = label2onehot(targets_L, batch_size_L, self.num_class)
        hard_pseudo_targets = label2onehot(hard_pseudo_targets, batch_size_U, self.num_class)

        # base fixmatch use hard pseudo labels, ABC use soft pseudo labels
        pseudo_targets = soft_pseudo_targets if mode == 'abc' else hard_pseudo_targets

        # generate mask for unlabeled data by threshold, double for two strong views

        mask_U_by_threshold = torch.cat([max_probs.ge(self.args.tau), max_probs.ge(self.args.tau)], dim=0).float()

        # generate mask for labeled data to be balanced

        weight_L_for_balance = self.generate_weight_L_for_balance(mode=mode,
                                                                  anneal=self.args.ada_weight_anneal_L,
                                                                  epoch=epoch,
                                                                  targets_L=targets_L,
                                                                  batch_size_L=batch_size_L)

        weight_U_for_balance = self.generate_weight_U_for_balance(mode=mode,
                                                                  anneal=self.args.ada_weight_anneal_U,
                                                                  epoch=epoch,
                                                                  meters=meters,
                                                                  distribution_type=self.args.ada_weight_type,
                                                                  hard_pseudo=hard_pseudo_targets)

        loss_L, loss_U = self.semi_loss(logits_L=logits_L, targets_L=targets_L,
                                        logits_U=torch.cat([logits_U_strong1, logits_U_strong2], dim=0),
                                        targets_U=torch.cat([pseudo_targets, pseudo_targets], dim=0),
                                        mask_L_for_balance=weight_L_for_balance,
                                        mask_U_for_balance=torch.cat([weight_U_for_balance, weight_U_for_balance],
                                                                     dim=0),
                                        mask_U_by_threshold=mask_U_by_threshold)

        return loss_L, loss_U, soft_pseudo_targets, weight_U_for_balance, weight_L_for_balance

    def generate_weight_L_for_balance(self, mode, epoch, anneal, targets_L, batch_size_L):
        if mode == 'abc':
            # 对于有标签数据，直接使用ground-truth作为类别分布
            stat_per_class = torch.tensor(self.sample_num_per_class['labeled']).float()
            weight_per_class = generate_probs_per_class(sample_fun_type=self.args.sample_fun_type,
                                                        stat_per_class=stat_per_class,
                                                        power=self.args.sample_power,
                                                        norm_type='local')
            if anneal:
                weight_per_class = 1 - (epoch / self.args.epochs) * (1 - weight_per_class)

            weight_L_for_balance = targets_L @ weight_per_class

        elif mode == 'base':
            weight_L_for_balance = torch.ones(batch_size_L).cuda().detach()
        else:
            raise KeyError()
        if self.args.no_mask_L_for_balance:
            weight_L_for_balance = torch.ones(batch_size_L).cuda().detach()

        return weight_L_for_balance.detach()

    def generate_weight_U_for_balance(self, mode, anneal, epoch, meters, hard_pseudo, distribution_type):
        # use every n epoch estimate
        if self.args.pd_distribution_estimate_nepoch > 0:
            select_pseudo_num_per_class = meters[f'select_pseudo_num_per_class_{mode}'].avg
            raw_pseudo_num_per_class = meters[f'raw_pseudo_num_per_class_{mode}'].avg
        # use ema estimate
        elif self.args.pd_stat_ema_momentum_wt > 0:
            select_pseudo_num_per_class = meters[f'select_pseudo_num_per_class_{mode}'].val
            raw_pseudo_num_per_class = meters[f'raw_pseudo_num_per_class_{mode}'].val
        else:
            raise KeyError

        if mode == 'abc':
            if distribution_type == 'no':
                stat_per_class = torch.ones(hard_pseudo.shape[1])
            elif distribution_type == 'gt':
                stat_per_class = torch.tensor(self.sample_num_per_class['labeled'])
            elif distribution_type == 'pd_select':
                stat_per_class = select_pseudo_num_per_class
            elif distribution_type == 'pd_raw':
                stat_per_class = raw_pseudo_num_per_class
            else:
                raise KeyError

            weight_per_class = generate_probs_per_class(sample_fun_type=self.args.sample_fun_type,
                                                        stat_per_class=stat_per_class,
                                                        power=self.args.sample_power,
                                                        norm_type='local', )

            if anneal:
                weight_per_class = 1 - (epoch / self.args.epochs) * (1 - weight_per_class)

            weight_U_for_balance = hard_pseudo @ weight_per_class

        elif mode == 'base':
            weight_U_for_balance = torch.ones(hard_pseudo.shape[0]).cuda().detach()
        else:
            raise KeyError()

        return weight_U_for_balance.detach()

    def get_batch_data(self, to_cuda=True):
        """get batch data from dataiter
            if to_cuda = True, move data to GPU, else keep on CPU
        """
        try:
            imgs_L, targets_L, indice_L = self.labeled_iter.next()
        except:
            self.labeled_iter = iter(self.labeled_loader)
            imgs_L, targets_L, indice_L = self.labeled_iter.next()
        try:
            (imgs_U_weak, imgs_U_strong1, imgs_U_strong2), targets_U, indice_U = self.unlabeled_iter.next()
        except:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            (imgs_U_weak, imgs_U_strong1, imgs_U_strong2), targets_U, indice_U = self.unlabeled_iter.next()

        if to_cuda:
            ret = {
                'labeled': (imgs_L.cuda(), targets_L.cuda(), indice_L),
                'unlabeled': (imgs_U_weak.cuda(), imgs_U_strong1.cuda(), imgs_U_strong2.cuda(), targets_U.cuda(), indice_U)
            }
        else:
            ret = {
                'labeled': (imgs_L, targets_L, indice_L),
                'unlabeled': (imgs_U_weak, imgs_U_strong1, imgs_U_strong2, targets_U, indice_U)
            }

        return ret
