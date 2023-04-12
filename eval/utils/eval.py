from __future__ import print_function, absolute_import
import numpy as np

__all__ = ['accuracy', 'shot_accuracy']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(correct[:k].size()[0] * correct[:k].size()[1]).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def shot_accuracy(correct_num_per_class: np.ndarray,
                  train_num_per_class,
                  test_num_per_class,
                  many_shot_thr=100,
                  low_shot_thr=20):
    """
    Args:
        correct_num_per_class: 每个类预测正确的样本数量，如果是dist_eval情况下，需要传入所有GPU上的sum
        train_num_per_class: 训练集中每个类别所含样本数量，用于区分many/medium/low shot
        test_num_per_class: 测试集上每个类别样本数量，与correct_num_per_class配合计算每个类别的准确率，无论dist_eval与否
            都是指的整个测试数据集中每个类别的样本数量
        many_shot_thr:
        low_shot_thr:
    """
    num_class = len(train_num_per_class)
    many_shot_acc = []
    median_shot_acc = []
    low_shot_acc = []
    acc_per_class = []

    for i in range(num_class):
        acc = (correct_num_per_class[i] / test_num_per_class[i]) * 100
        acc_per_class.append(acc)
        if train_num_per_class[i] >= many_shot_thr:
            many_shot_acc.append(acc)
        elif train_num_per_class[i] <= low_shot_thr:
            low_shot_acc.append(acc)
        else:
            median_shot_acc.append(acc)

    ret = {
        'acc_per_class': np.array(acc_per_class),
        'all_acc': np.mean(acc_per_class),
        'many_shot_acc': np.mean(many_shot_acc),
        'medium_shot_acc': np.mean(median_shot_acc),
        'low_shot_acc': np.mean(low_shot_acc)
    }

    return ret
