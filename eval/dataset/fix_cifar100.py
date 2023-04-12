import numpy as np
from PIL import Image

import torchvision
from .utils import train_split_labeled_unlabeled, TransformTwice, get_transforms

# Parameters for data
cifar100_mean = (0.5071, 0.4867, 0.4408)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2675, 0.2565, 0.2761)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def get_cifar100(
        root,
        labeled_sample_num_per_class,
        unlabeled_sample_num_per_class,
        args,
        download=True):
    """
    Args:
        root: root path to the dataset dir
        labeled_sample_num_per_class: sample nums per class in labeled dataset
        unlabeled_sample_num_per_class: sample nums per class in unlabeled dataset
        args:
        download:
    """
    base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)

    transform_weak, transform_strong, transform_val = get_transforms(cifar100_mean, cifar100_std)

    train_labeled_idxs, train_unlabeled_idxs = train_split_labeled_unlabeled(base_dataset.targets,
                                                                             labeled_sample_num_per_class,
                                                                             unlabeled_sample_num_per_class,
                                                                             num_class=100,
                                                                             seed=args.seed)

    train_labeled_dataset = CIFAR100_labeled(root=root,
                                             indice=train_labeled_idxs,
                                             train=True,
                                             transform=transform_weak)

    train_unlabeled_dataset = CIFAR100_unlabeled(root=root,
                                                 indice=train_unlabeled_idxs,
                                                 train=True,
                                                 transform=TransformTwice(transform_weak, transform_strong))

    val_dataset = CIFAR100_labeled(root=root,
                                   train=False,
                                   transform=transform_val,
                                   download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


class CIFAR100_labeled(torchvision.datasets.CIFAR100):
    def __init__(self,
                 root,
                 indice=None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):

        super(CIFAR100_labeled, self).__init__(root,
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


class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self,
                 root,
                 indice,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(CIFAR100_unlabeled, self).__init__(root,
                                                 indice,
                                                 train=train,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 download=download)
