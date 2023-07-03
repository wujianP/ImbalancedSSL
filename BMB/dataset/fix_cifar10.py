import numpy as np
from PIL import Image

import torchvision
from .utils import train_split_labeled_unlabeled, TransformTwice, get_transforms

# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def get_cifar10(root,
                labeled_sample_num_per_class,
                unlabeled_sample_num_per_class,
                args,
                download=True):
    """
    Args:
        root: root path to the dataset dir
        labeled_sample_num_per_class: sample nums per class in labeled dataset
        unlabeled_sample_num_per_class: sample nums per class in unlabeled dataset
        args
        download:
    """
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)

    train_labeled_idxs, train_unlabeled_idxs = train_split_labeled_unlabeled(base_dataset.targets,
                                                                             labeled_sample_num_per_class,
                                                                             unlabeled_sample_num_per_class,
                                                                             num_class=10,
                                                                             seed=args.seed)

    transform_weak, transform_strong, transform_val = get_transforms(cifar10_mean, cifar10_std)

    train_labeled_dataset = CIFAR10_labeled(root,
                                            indice=train_labeled_idxs,
                                            train=True,
                                            transform=transform_weak)

    train_unlabeled_dataset = CIFAR10_unlabeled(root,
                                                indice=train_unlabeled_idxs,
                                                train=True,
                                                transform=TransformTwice(transform_weak, transform_strong))
    val_dataset = CIFAR10_labeled(root,
                                  train=False,
                                  transform=transform_val,
                                  download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


class CIFAR10_labeled(torchvision.datasets.CIFAR10):
    def __init__(self,
                 root,
                 indice=None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(CIFAR10_labeled, self).__init__(root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)
        if indice is not None:
            self.data = self.data[indice]
            self.targets = np.array(self.targets)[indice]

        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self,
                 root,
                 indice,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(CIFAR10_unlabeled, self).__init__(root,
                                                indice,
                                                train=train,
                                                transform=transform,
                                                target_transform=target_transform,
                                                download=download)
