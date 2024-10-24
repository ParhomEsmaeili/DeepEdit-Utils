import argparse
from os.path import dirname as up
import os
import sys
# file_dir = up(up(up(os.path.abspath(__file__))))
# sys.path.append(os.path.join(file_dir, 'nnUnet', 'nnUNet','nnunetv2', 'dataset_conversion'))
from nnUnet_generate_dataset_json import generate_dataset_json
import json 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", default='0')
    parser.add_argument("--train_folds", nargs="+", default=['1'])
    parser.add_argument("--my_json_path", default='/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/train_val_split_dataset.json')
    
    args = parser.parse_args() 

    train_folds_to_string = '_'.join(args.train_folds)

    input_output_folder = f'/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_{train_folds_to_string}'
    
    input_channel_names = {
        0:'T2'
    }

    input_labels = {
        'background': 0,
        'tumor': 1
    }
    
    input_num_training_cases = 0
    with open(args.my_json_path) as f:
        full_dict = json.load(f)
        # train_files = np.array(full_dict["training"])
    
    for i in args.train_folds:
        sub_fold_list = full_dict[f'fold_{i}']
        input_num_training_cases += len(sub_fold_list)
    
    input_num_training_cases += len(full_dict[f'fold_{args.val_fold}'])


    generate_dataset_json(output_folder=input_output_folder,
                          channel_names=input_channel_names, 
                          labels=input_labels,
                          num_training_cases=input_num_training_cases,
                          file_ending='.nii.gz')
    