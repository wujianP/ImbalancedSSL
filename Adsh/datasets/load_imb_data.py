import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision import datasets
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

cifar100_mean = (0.5071, 0.4867, 0.4408)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2675, 0.2565, 0.2761)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_strong_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])
transform_strong_cifar100.transforms.insert(0, RandAugment(3, 4))
transform_strong_cifar100.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])


class TransformFixMatch(object):
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, x):
        weak = self.transform(x)
        strong1 = self.transform2(x)
        return weak, strong1


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None):
        super().__init__(root, train=train,
                         transform=transform)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None):
        super().__init__(root, train=train,
                         transform=transform)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_imb_data(num_max, num_classes, imb_ratio):
    mu = np.power(1 / imb_ratio, 1 / (num_classes - 1))
    class_num_list = []
    for i in range(num_classes):
        if i == (num_classes - 1):
            class_num_list.append(int(num_max / imb_ratio))
        else:
            class_num_list.append(int(num_max * np.power(mu, i)))
    return list(class_num_list)


def l_u_split(args, labels, label_per_class, unlabel_per_class):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []

    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        if args.manualSeed !=0:
            np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class[i]])
        unlabeled_idx.extend(idx[:label_per_class[i] + unlabel_per_class[i]])

    return labeled_idx, unlabeled_idx


def get_imb_cifar10(args, root):

    dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max * args.label_ratio, args.num_classes, args.imb_ratio_u)

    print("f#Labeled : ", N_SAMPLES_PER_CLASS)
    print("f#Unlabeled: ", U_SAMPLES_PER_CLASS)

    labeled_idx, unlabeled_idx = l_u_split(
        args, dataset.targets, N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    labeled_dataset = CIFAR10SSL(
        root, labeled_idx, train=True, transform=transform_train)

    unlabeled_dataset = CIFAR10SSL(
        root, unlabeled_idx, train=True,
        transform=TransformFixMatch(transform_train, transform_strong))

    test_dataset = torchvision.datasets.CIFAR10(
        root, train=False, transform=transform_val, download=True)

    return N_SAMPLES_PER_CLASS, labeled_dataset, unlabeled_dataset, test_dataset


def get_imb_cifar100(args, root):

    dataset = torchvision.datasets.CIFAR100(root, train=True, download=True)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max * args.label_ratio, args.num_classes, args.imb_ratio_u)

    print("f#Labeled : ", N_SAMPLES_PER_CLASS)
    print("f#Unlabeled: ", U_SAMPLES_PER_CLASS)

    labeled_idx, unlabeled_idx = l_u_split(
        args, dataset.targets, N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    labeled_dataset = CIFAR100SSL(
        root, labeled_idx, train=True, transform=transform_train_cifar100)

    unlabeled_dataset = CIFAR100SSL(
        root, unlabeled_idx, train=True,
        transform=TransformFixMatch(transform_train_cifar100, transform_strong_cifar100))

    test_dataset = torchvision.datasets.CIFAR100(
        root, train=False, transform=transform_val, download=True)

    return N_SAMPLES_PER_CLASS, labeled_dataset, unlabeled_dataset, test_dataset


DATASET_GETTERS = {
    'cifar10': get_imb_cifar10,
    'cifar100': get_imb_cifar100,
}
