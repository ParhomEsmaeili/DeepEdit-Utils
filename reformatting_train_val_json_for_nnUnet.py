import os
import shutil
import argparse 
import json
import sys 
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# alist=[
#     "something1",
#     "something12",
#     "something17",
#     "something2",
#     "something25",
#     "something29"]

# alist.sort(key=natural_keys)
# print(alist)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_split_json", default="/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_MSD_Format/train_val_split_dataset.json")
    parser.add_argument("--val_fold", default='0')
    parser.add_argument("--train_folds", nargs="+", default=['1','2','3','4'])

    args = parser.parse_args()

    train_folds_to_string = '_'.join(args.train_folds)
    # input_folder_path = '/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_MSD_Format'
    output_folder_path = f'/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_{train_folds_to_string}_SPLIT_JSON'

    with open(args.fold_split_json) as f:
        full_dict = json.load(f)

    train_list = []
    for train_fold in args.train_folds:
        train_fold_dict = full_dict[f'fold_{train_fold}']
        for image_dict in train_fold_dict:
            image_name = image_dict["image"].split('/')[-1]
            train_list.append(image_name)
    #sort by numeric value
    train_list.sort(key=natural_keys)


    val_list = []
    val_fold_dict = full_dict[f'fold_{args.val_fold}']

    for image_dict in val_fold_dict:
        image_name = image_dict["image"].split('/')[-1]
        val_list.append(image_name)
    
    #sort by numeric value 

    val_list.sort(key=natural_keys)

    train_validation_split_dict_nnunetformat = dict()
    #putting the configuration into the nn unet format
    train_validation_split_dict_nnunetformat["train"] = train_list 
    train_validation_split_dict_nnunetformat["val"] = val_list 

    #putting all of the dicts into a list together 

    final_list = [train_validation_split_dict_nnunetformat, train_validation_split_dict_nnunetformat]


    with open(os.path.join(output_folder_path, 'splits_final.json'), 'w') as f:
        json.dump(final_list, f)
