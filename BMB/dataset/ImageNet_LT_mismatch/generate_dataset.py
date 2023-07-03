import os
import numpy as np
import argparse


def make_imbalanced_data(max_num, imb_ratio, class_num=1000):
    """
    根据给定参数确定每个类别的样本数量

    :param max_num: int，每个类别中最多样本数量
    :param imb_ratio: int，表示长尾程度的因子
    :param class_num: int，类别数量
    :return: list，每个类别的样本数量
    """
    mu = np.power(1 / imb_ratio, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / imb_ratio))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    return list(class_num_list)


def save_files(path_list, label_list, save_path):
    with open(save_path, 'w') as f:
        print(len(path_list))
        assert len(path_list) == len(label_list)
        for i in range(len(path_list)):
            line = f"{path_list[i]} {label_list[i]}\n"
            f.write(line)
    f.close()


def sample_images(data_path, max_num_l, max_num_u, imb_ratio_l, imb_ratio_u, seed=0, save_path=None):
    """
    在给定路径下采样图片，并将结果保存到文件
    """
    # 获取所有类别的名称和对应的编号
    class_names = sorted(os.listdir(data_path))
    assert len(class_names) == 1000
    class_dict = {class_names[i]: i for i in range(len(class_names))}

    # 为有标签数据集和无标签数据集各自确定每个类别的样本数量
    sample_num_per_class_labeled = make_imbalanced_data(max_num_l, imb_ratio_l, len(class_names))
    sample_num_per_class_unlabeled = make_imbalanced_data(max_num_u, imb_ratio_u, len(class_names))

    print(f'sample_num_per_class_labeled: {sample_num_per_class_labeled}')
    print(f'sample_num_per_class_unlabeled: {sample_num_per_class_unlabeled}')
    print(sum(sample_num_per_class_labeled))
    print(sum(sample_num_per_class_unlabeled))

    save_files(sample_num_per_class_labeled,
               [sample_num_per_class_unlabeled[i] + sample_num_per_class_labeled[i]
                for i in range(len(sample_num_per_class_labeled))],
               save_path=os.path.join(save_path,
                                      f'maxL{max_num_l}_maxU{max_num_u}_imbL'
                                      f'{imb_ratio_l}_imbU{imb_ratio_u}_sampleNum.txt'))

    # 用于存储采样到的样本的路径和标签信息
    labeled_path_list, labeled_label_list = [], []
    unlabeled_path_list, unlabeled_label_list = [], []

    # 对于每个类别，分别在有标签和无标签数据集中进行采样
    for class_name in class_names:
        # 读取当前类别下的所有样本路径
        class_path = os.path.join(data_path, class_name)
        all_image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        # 随机打乱样本路径的顺序
        if seed != 0:
            np.random.shuffle(all_image_paths)
        class_idx = class_dict[class_name]
        sample_num_labeled = sample_num_per_class_labeled[class_idx]
        sample_num_unlabeled = sample_num_per_class_unlabeled[class_idx]
        assert len(all_image_paths) >= (sample_num_unlabeled + sample_num_labeled)
        labeled_path_list.extend(all_image_paths[:sample_num_labeled])
        unlabeled_path_list.extend(all_image_paths[:sample_num_labeled + sample_num_unlabeled])
        labeled_label_list.extend([class_idx] * sample_num_labeled)
        unlabeled_label_list.extend([class_idx] * (sample_num_labeled + sample_num_unlabeled))

    return labeled_path_list, labeled_label_list, unlabeled_path_list, unlabeled_label_list


def generate_dataset(data_path, save_path, max_num_l, max_num_u, imb_ratio_l, imb_ratio_u):
    labeled_path_list, labeled_label_list, unlabeled_path_list, unlabeled_label_list = sample_images(
        data_path=data_path,
        max_num_l=max_num_l,
        max_num_u=max_num_u,
        imb_ratio_l=imb_ratio_l,
        imb_ratio_u=imb_ratio_u,
        save_path=save_path
    )

    save_path_labeled = os.path.join(save_path, f'maxL{max_num_l}_maxU{max_num_u}_imbL{imb_ratio_l}_imbU{imb_ratio_u}_labeled.txt')
    save_path_unlabeled = os.path.join(save_path, f'maxL{max_num_l}_maxU{max_num_u}_imbL{imb_ratio_l}_imbU{imb_ratio_u}_unlabeled.txt')
    save_files(path_list=labeled_path_list, label_list=labeled_label_list, save_path=save_path_labeled)
    save_files(path_list=unlabeled_path_list, label_list=unlabeled_label_list, save_path=save_path_unlabeled)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate semi-imbalanced ImageNet dataset')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--max_num_l', type=int)
    parser.add_argument('--max_num_u', type=int)
    parser.add_argument('--imb_ratio_l', type=int)
    parser.add_argument('--imb_ratio_u', type=int)

    args = parser.parse_args()

    generate_dataset(
        data_path=args.data_path,
        save_path=args.save_path,
        max_num_l=args.max_num_l,
        max_num_u=args.max_num_u,
        imb_ratio_l=args.imb_ratio_l,
        imb_ratio_u=args.imb_ratio_u
    )

    print(f'Generated, saved at: {args.save_path}')
