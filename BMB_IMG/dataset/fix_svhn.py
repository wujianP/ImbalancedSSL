import numpy as np
from PIL import Image

import torchvision
from .utils import train_split_labeled_unlabeled, TransformTwice, get_transforms

# Parameters for data
SVHN_mean = (0.4377, 0.4438, 0.4728)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
SVHN_std = (0.1980, 0.2010, 0.1970)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def get_svhn(root,
             labeled_sample_num_per_class,
             unlabeled_sample_num_per_class,
             download=True):
    base_train_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
    base_test_dataset = torchvision.datasets.SVHN(root, split='test', download=download)

    test_idxs = test_split(base_test_dataset.labels)
    train_labeled_idxs, train_unlabeled_idxs = train_split_labeled_unlabeled(base_train_dataset.labels,
                                                                             labeled_sample_num_per_class,
                                                                             unlabeled_sample_num_per_class,
                                                                             num_class=10)

    transform_weak, transform_strong, transform_val = get_transforms(SVHN_mean, SVHN_std)

    train_labeled_dataset = SVHN_labeled(root,
                                         train_labeled_idxs,
                                         split='train',
                                         transform=transform_weak)

    train_unlabeled_dataset = SVHN_unlabeled(root,
                                             train_unlabeled_idxs,
                                             split='train',
                                             transform=TransformTwice(transform_weak, transform_strong))
    test_dataset = SVHN_labeled(root,
                                test_idxs,
                                split='test',
                                transform=transform_val,
                                download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def test_split(labels):
    labels = np.array(labels)
    test_idxs = []
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:1500])
    np.random.shuffle(test_idxs)
    return test_idxs


def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])


class SVHN_labeled(torchvision.datasets.SVHN):
    def __init__(self,
                 root,
                 indice=None,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=True):

        super(SVHN_labeled, self).__init__(root,
                                           split=split,
                                           transform=transform,
                                           target_transform=target_transform,
                                           download=download)
        if indice is not None:
            self.data = self.data[indice]
            self.labels = np.array(self.labels)[indice]

        self.data = transpose(self.data)
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHN_unlabeled(SVHN_labeled):
    def __init__(self,
                 root,
                 indice,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=True):

        super(SVHN_unlabeled, self).__init__(root,
                                             indice,
                                             split=split,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)
