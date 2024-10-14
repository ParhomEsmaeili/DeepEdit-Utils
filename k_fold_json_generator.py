import json
import argparse 
import numpy as np
import os


def parse_arguments():
    parser = argparse.ArgumentParser("Data Preprocessing")
    parser.add_argument("--base_dir", default="/home/parhomesmaeili/Radiology_Datasets")
    parser.add_argument("--studies", default="BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2")
    parser.add_argument("--num_folds", default='5')

    return parser 



if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    dataset_dir = os.path.join(args.base_dir, args.studies)
    
    with open(os.path.join(dataset_dir, "dataset.json")) as f:
        full_dict = json.load(f)
        train_files = np.array(full_dict["training"])
    
    np.random.shuffle(train_files)
    
    num_imgs = len(train_files)
    imgs_per_fold = num_imgs // int(args.num_folds)

    full_nested_list = []
    for i in range(int(args.num_folds)):
        print(i)
        if i == int(args.num_folds) - 1:
            sublist = train_files[i * imgs_per_fold:]
        else: 
            sublist = train_files[i * imgs_per_fold: i * imgs_per_fold + imgs_per_fold]
        full_nested_list.append(sublist)
    
    print(full_nested_list)

    output_dict = dict()
    for i in range(int(args.num_folds)):
        output_dict[f"fold_{i}"] = full_nested_list[i].tolist()
    
    with open(os.path.join(dataset_dir, "train_val_split_dataset.json"), 'w') as f:
        json.dump(output_dict, f)



