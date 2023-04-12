import os
import numpy as np
import math

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

        labeled_sample_num_per_class.append(math.ceil(len(index) * labeled_ratio))   # 当前类有标签样本的数量
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
            f.write(labeled_path[i]+' '+str(labeled_labels[i])+'\n')
        f.close()
    with open(f'{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_unlabeled.txt', 'w') as f:
        assert len(unlabeled_path) == len(unlabeled_labels)
        for i in range(len(unlabeled_path)):
            f.write(unlabeled_path[i]+' '+str(unlabeled_labels[i])+'\n')
        f.close()
    with open(f'{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_sample_num.txt', 'w') as f:
        # 第一列是labeled, 第二列是unlabeled
        assert len(labeled_sample_num_per_class) == len(unlabeled_sample_num_per_class)
        for i in range(len(labeled_sample_num_per_class)):
            f.write(str(labeled_sample_num_per_class[i])+' '+str(unlabeled_sample_num_per_class[i])+'\n')
        f.close()


if __name__ == '__main__':
    generate_imagenetLT_train_split_annotation_file(train_annotation_file='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT\ImageNet_LT_train.txt',
                                                    save_file_root='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT',
                                                    labeled_ratio=20,
                                                    num_class=1000)
    generate_imagenetLT_train_split_annotation_file(train_annotation_file='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT\ImageNet_LT_train.txt',
                                                    save_file_root='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT',
                                                    labeled_ratio=50,
                                                    num_class=1000)
    generate_imagenetLT_train_split_annotation_file(train_annotation_file='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT\ImageNet_LT_train.txt',
                                                    save_file_root='D:\我的文件\codes\imporvedABC\dataset\ImageNet_LT',
                                                    labeled_ratio=80,
                                                    num_class=1000)
