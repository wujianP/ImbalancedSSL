import math

import numpy as np
import os
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault


__all__ = ['train_split_labeled_unlabeled', 'TransformTwice', 'get_transforms',
           'imagenet_train_split_labeled_unlabeled', 'imagenet_get_transforms', 'imagenet127_get_transforms',
           'make_imbalanced_data', 'generate_imagenetLT_train_split_annotation_file']


def make_imbalanced_data(max_num, class_num, imb_ratio, imb_type):
    """
    make imbalanced data
    Args:
        max_num: the number of samples in the max class
        class_num: how many classes in the dataset
        imb_ratio: imbalanced ration = max_num/min_num
        imb_type: imbalanced data type include ['step', 'long']
    Return:
        class_num_list: (list) number of samples in different classes,
        sorted in non-ascending order.
    """
    assert imb_type in ['long', 'step'], "Unknown imbalance type"

    if imb_type == 'long':
        mu = np.power(1 / imb_ratio, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / imb_ratio))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))

    elif imb_type == 'step':
        class_num_list = []
        for i in range(class_num):
            if i < int(class_num / 2):
                # majority classes
                class_num_list.append(int(max_num))
            else:
                # minority classes
                class_num_list.append(int(max_num / imb_ratio))

    else:
        raise KeyError

    return list(class_num_list)


def train_split_labeled_unlabeled(labels, n_labeled_per_class, n_unlabeled_per_class, num_class, seed=0):

    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        if seed != 0:
            np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i]+n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


def generate_imagenetLT_train_split_annotation_file(train_annotation_file,
                                                    save_file_root,
                                                    labeled_ratio,
                                                    num_class=1000):
    """
    save_file_root: root to the dir where to save the annotation files
    train_annotation_file: path to the .txt file (train), for the entire annotations(fully supervised)
    """
    labeled_ratio /= 100

    all_path = []   # 所有样本的path，包括labeled/unlabelled
    all_labels = []   # 所有样本的label，包括labeled/unlabelled
    with open(train_annotation_file) as f:
        for line in f:
            all_path.append(os.path.join(line.split()[0]))
            all_labels.append(int(line.split()[1]))
    all_path = np.array(all_path)
    all_labels = np.array(all_labels)

    index_per_class = [[] for _ in range(num_class)]    # 每个类样本在all当中的index
    for i in range(len(all_labels)):
        class_label = int(all_labels[i])
        index_per_class[class_label].append(i)

    labeled_sample_num_per_class = []    # 每个类别中labeled样本数量
    unlabeled_sample_num_per_class = []  # 每个类别中unlabelled样本数量

    labeled_path, labeled_labels = [], []   # labeled样本的path和label(所有类别)
    unlabeled_path, unlabeled_labels = [], []   # unlabeled样本的path和label(所有类别)
    for i in range(num_class):
        index = index_per_class[i]  # 当前类所包含的样本对应的all中的index

        labeled_sample_num_per_class.append(int(len(index) * labeled_ratio))   # 当前类有标签样本的数量
        unlabeled_sample_num_per_class.append(len(index))   # 当前类无标签样本的数量,使用所有的数据作为无标签样本

        labeled_index = index[:labeled_sample_num_per_class[i]]    # 当前类中labeled样本在all中index
        unlabeled_index = index[:]  # 当前类中unlabelled样本在all中的index

        labeled_path.extend(all_path[labeled_index])
        labeled_labels.extend(all_labels[labeled_index])

        unlabeled_path.extend(all_path[unlabeled_index])
        unlabeled_labels.extend(all_labels[unlabeled_index])

    # 划分结果写入annotation file
    with open(f'{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_labeled.txt', 'w') as f:
        assert len(labeled_path) == len(labeled_labels)
        for i in range(len(labeled_path)):
            f.write(labeled_path[i]+' '+labeled_labels[i]+'\n')
        f.close()
    with open(f'{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_unlabeled.txt', 'w') as f:
        assert len(unlabeled_path) == len(unlabeled_labels)
        for i in range(len(unlabeled_path)):
            f.write(unlabeled_path[i]+' '+unlabeled_labels[i]+'\n')
        f.close()
    with open(f'{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_sample_num.txt', 'w') as f:
        # 第一列是labeled, 第二列是unlabeled
        assert len(labeled_sample_num_per_class) == len(unlabeled_sample_num_per_class)
        for i in range(len(labeled_sample_num_per_class)):
            f.write(str(labeled_sample_num_per_class[i])+' '+str(unlabeled_sample_num_per_class[i])+'\n')
        f.close()


def imagenet_train_split_labeled_unlabeled(root, train_annotation_file, labeled_ratio, num_class=1000):
    """
    root: root to the dataset dir
    train_annotation_file: path to the .txt file (train)
    """
    labeled_ratio /= 100

    # # # # # # # # # # # # # # # # 划分labeled和unlabelled数据集 # # # # # # # # # # # # # # # #

    all_path = []   # 所有样本的path，包括labeled/unlabelled
    all_labels = []   # 所有样本的label，包括labeled/unlabelled
    with open(train_annotation_file) as f:
        for line in f:
            all_path.append(os.path.join(root, line.split()[0]))
            all_labels.append(int(line.split()[1]))
    all_path = np.array(all_path)
    all_labels = np.array(all_labels)

    index_per_class = [[] for _ in range(num_class)]    # 每个类样本在all当中的index
    for i in range(len(all_labels)):
        class_label = int(all_labels[i])
        index_per_class[class_label].append(i)

    labeled_sample_num_per_class = []    # 每个类别中labeled样本数量
    unlabeled_sample_num_per_class = []  # 每个类别中unlabelled样本数量

    labeled_path, labeled_labels = [], []   # labeled样本的path和label(所有类别)
    unlabeled_path, unlabeled_labels = [], []   # unlabeled样本的path和label(所有类别)
    for i in range(num_class):
        index = index_per_class[i]  # 当前类所包含的样本对应的all中的index

        labeled_sample_num_per_class.append(math.ceil(len(index) * labeled_ratio))   # 当前类有标签样本的数量
        unlabeled_sample_num_per_class.append(len(index) - int(len(index) * labeled_ratio))   # 当前类无标签样本的数量

        labeled_index = index[:labeled_sample_num_per_class[i]]    # 当前类中labeled样本在all中index
        unlabeled_index = index[labeled_sample_num_per_class[i]:]  # 当前类中unlabelled样本在all中的index

        labeled_path.extend(all_path[labeled_index])
        labeled_labels.extend(all_labels[labeled_index])

        unlabeled_path.extend(all_path[unlabeled_index])
        unlabeled_labels.extend(all_labels[unlabeled_index])

    sample_num_per_class = {
        'labeled': labeled_sample_num_per_class,
        'unlabeled': unlabeled_sample_num_per_class
    }

    return labeled_path, labeled_labels, unlabeled_path, unlabeled_labels, sample_num_per_class


def get_transforms(mean, std):
    transform_weak = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        RandAugment(3, 4),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        CutoutDefault(16)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_weak, transform_strong, transform_val


def imagenet127_get_transforms(mean, std, crop_size):
    transform_weak = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=int(crop_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=int(crop_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(int(crop_size / 2)))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # # img_size = int(crop_size/0.875)
    # crop_size = crop_size
    # transform_weak = transforms.Compose([
    #     # transforms.Resize(img_size),
    #     # transforms.RandomCrop(crop_size),
    #     transforms.RandomCrop(crop_size, padding=int(crop_size / 8)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    #
    # transform_strong = transforms.Compose([
    #     # transforms.Resize(img_size),
    #     RandAugment(3, 4),
    #     # transforms.RandomCrop(crop_size),
    #     transforms.RandomCrop(crop_size, padding=int(crop_size / 8)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    #     CutoutDefault(crop_size/2)
    # ])
    #
    # transform_val = transforms.Compose([
    #     # transforms.Resize(img_size),
    #     # transforms.CenterCrop(crop_size),
    #     transforms.RandomCrop(crop_size, padding=int(crop_size / 8)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])

    return transform_weak, transform_strong, transform_val


def imagenet_get_transforms(mean, std):
    transform_weak = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(256),
        RandAugment(3, 4),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # CutoutDefault(64)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_weak, transform_strong, transform_val


class TransformTwice:
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, inp):
        out_weak = self.transform_weak(inp)
        out_strong1 = self.transform_strong(inp)
        out_strong2 = self.transform_strong(inp)

        return out_weak, out_strong1, out_strong2


