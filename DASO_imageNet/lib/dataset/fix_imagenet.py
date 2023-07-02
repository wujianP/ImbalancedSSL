from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms
from RandAugment import RandAugment

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


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
        # out_strong2 = self.transform_strong(inp)

        return out_weak, out_strong1


def build_imagenet_dataset(root,
                           annotation_file_train_labeled,
                           annotation_file_train_unlabeled,
                           annotation_file_val,
                           num_per_class):
    sample_num_per_class = {'labeled': [], 'unlabeled': []}
    with open(num_per_class) as f:
        for line in f:
            sample_num_per_class['labeled'].append(int(line.split()[0]))
            sample_num_per_class['unlabeled'].append(int(line.split()[1]))
        f.close()

    # get annotation for labeled train data
    labeled_path, labeled_labels = [], []
    with open(annotation_file_train_labeled) as f:
        for line in f:
            labeled_path.append(os.path.join(root, line.split()[0]))
            labeled_labels.append(int(line.split()[1]))
        f.close()

    # get annotation for unlabeled train data
    unlabeled_path, unlabeled_labels = [], []
    with open(annotation_file_train_unlabeled) as f:
        for line in f:
            unlabeled_path.append(os.path.join(root, line.split()[0]))
            unlabeled_labels.append(int(line.split()[1]))
        f.close()

    # get annotation file for val
    val_path, val_labels = [], []
    with open(annotation_file_val) as f:
        for line in f:
            val_path.append(os.path.join(root, line.split()[0]))
            val_labels.append(int(line.split()[1]))
        f.close()

    # get transform
    transform_weak, transform_strong, transform_val = imagenet_get_transforms(imagenet_mean, imagenet_std)

    # construct labeled and unlabeled dataset
    train_labeled_dataset = ImageNetLT(path_list=labeled_path,
                                       label_list=labeled_labels,
                                       transform=transform_weak,
                                       train=True)

    train_unlabeled_dataset = ImageNetLT(path_list=unlabeled_path,
                                         label_list=unlabeled_labels,
                                         transform=TransformTwice(transform_weak, transform_strong),
                                         train=True)
    val_dataset = ImageNetLT(path_list=val_path,
                             label_list=val_labels,
                             transform=transform_val,
                             train=False)

    print(f"#Labeled: {len(labeled_labels)} #Unlabeled: {len(unlabeled_labels)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, val_dataset
    # return sample_num_per_class, train_labeled_dataset, train_unlabeled_dataset, val_dataset


class ImageNetLT(Dataset):
    def __init__(self, path_list, label_list, transform=None, train=True):
        self.path_list = path_list
        self.targets = label_list
        self.transform = transform
        self.num_classes = 1000
        self.train = train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.path_list[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        f.close()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index
