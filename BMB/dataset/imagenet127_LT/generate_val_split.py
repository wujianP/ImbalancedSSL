import random
import os

random.seed(2023)

def gen(val_ann_path, save_file_root, val_folder):
    folder_list = os.listdir(val_folder)
    name2folder = {}
    for fname in folder_list:
        for name in os.listdir(os.path.join(val_folder, fname)):
            name2folder[name] = fname
    
    anno_dict = {}
    balance_anno_dict = {}
    
    with open(val_ann_path, 'r') as f:
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
        idx = list(range(num))
        random.shuffle(idx)
        idx = sorted(idx)[:50]
        balance_anno_dict[label] = [anno_list[i] for i in idx]
        
        balance_anno_dict[label] = anno_list
    
    with open(os.path.join(save_file_root, 'ImageNet127_LT_val.txt'), 'w') as f:
        for label in balance_anno_dict:
            filelist = balance_anno_dict[label]
            for filepath in filelist:
                fp = filepath.split(' ')[0]
                f.write('val/%s/'%(name2folder[fp]))
                f.write(filepath.strip())
                f.write('\n')
    
save_file_root = '/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT'
val_ann_path = '/discobox/jia/dataset/imagenet/wmigftl/label_sets/imagenet127_LT/val_up_down_127.txt'
gen(val_ann_path, save_file_root, '/dev/shm/imagenet/val')

# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_labeled.txt'
# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_unlabeled.txt'
# '{save_file_root}/ImageNet_LT_train_semi_{int(labeled_ratio*100)}_sample_num.txt'


