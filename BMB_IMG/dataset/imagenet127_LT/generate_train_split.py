import random
import os

random.seed(2023)

def gen(train_ann_path, labeled_ratio, save_file_root):
    anno_dict = {}
    labeled_dict = {}
    unlabeled_dict = {}
    sample_num_dict = {}

    with open(train_ann_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cls = int(line.split(' ')[1])

            if cls not in anno_dict:
                anno_dict[cls] = []

            anno_dict[cls].append(line)
    
    label_list = sorted(anno_dict)
    for label in label_list:
        anno_list = anno_dict[label]
        num = len(anno_list)
        sample_num_dict[label] = [int(num * labeled_ratio / 100), num]
        labeled_dict[label] = random.sample(anno_list, int(num * labeled_ratio / 100)) 
        unlabeled_dict[label] = anno_list
    
    with open(os.path.join(save_file_root, 'ImageNet127_LT_train_semi_%d_sample_num.txt'%labeled_ratio), 'w') as f:
        for label in sample_num_dict:
            f.write('%d %d'%(sample_num_dict[label][0], sample_num_dict[label][1]))
            f.write('\n')
 
    with open(os.path.join(save_file_root, 'ImageNet127_LT_train_semi_%d_labeled.txt'%labeled_ratio), 'w') as f:
        for label in labeled_dict:
            filelist = labeled_dict[label]
            for filepath in filelist:
                f.write('train/')
                f.write(filepath.strip())
                f.write('\n')
    
    with open(os.path.join(save_file_root, 'ImageNet127_LT_train_semi_%d_unlabeled.txt'%labeled_ratio), 'w') as f:
        for label in unlabeled_dict:
            filelist = unlabeled_dict[label]
            for filepath in filelist:
                f.write('train/')
                f.write(filepath.strip())
                f.write('\n')
    

labeled_ratio = 10
save_file_root = '/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT'
train_ann_path = '/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT/train_up_down_127.txt'
gen(train_ann_path, labeled_ratio, save_file_root)

# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_labeled.txt'
# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_unlabeled.txt'
# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_sample_num.txt'


