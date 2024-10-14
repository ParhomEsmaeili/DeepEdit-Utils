import os
import shutil
import argparse 
import json
import sys 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_split_json", default="/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_MSD_Format/train_val_split_dataset.json")
    parser.add_argument("--val_fold", default='0')
    parser.add_argument("--train_folds", nargs="+", default=['1'])

    args = parser.parse_args()

    train_folds_to_string = '_'.join(args.train_folds)
    input_folder_path = '/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_1_2_3_4'
    output_folder_path = f'/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_{train_folds_to_string}'

    with open(args.fold_split_json) as f:
        full_dict = json.load(f)

    

    #splitting the fold collection into the sub-chunks that need to be moved into the training dataset. 

    #this is the sum of the train folds and val fold. 
    image_folds_transferred = args.train_folds + [args.val_fold]

    for fold in image_folds_transferred:
        extract_fold_sublist = full_dict[f'fold_{fold}']
        for sub_dict in extract_fold_sublist:
            deepeditplusplus_version_image_name = sub_dict['image'].split('/')[-1]
            nnUnet_version = deepeditplusplus_version_image_name + '_0000.nii.gz'

            os.makedirs(os.path.join(output_folder_path, 'imagesTr'), exist_ok=True)
            shutil.copy(os.path.join(input_folder_path, 'imagesTr', nnUnet_version), os.path.join(output_folder_path, 'imagesTr', nnUnet_version))


    # imagesTr_path = os.path.join(args.dataset_path_directory, 'imagesTr')

    for fold in image_folds_transferred:
        extract_fold_sublist = full_dict[f'fold_{fold}']
        for sub_dict in extract_fold_sublist:
            deepeditplusplus_version_image_name = sub_dict['image'].split('/')[-1] + '.nii.gz'
            # nnUnet_version = deepeditplusplus_version_image_name + '_0000.nii.gz'

            os.makedirs(os.path.join(output_folder_path, 'labelsTr'), exist_ok=True)
            shutil.copy(os.path.join(input_folder_path, 'labelsTr', deepeditplusplus_version_image_name), os.path.join(output_folder_path, 'labelsTr', deepeditplusplus_version_image_name))


    # imagesTr_path = os.path.join(args.dataset_path_directory, 'imagesTr')

    #copying over the imagesTs 
    shutil.copytree(os.path.join(input_folder_path, 'imagesTs'), os.path.join(output_folder_path, 'imagesTs'))

    #copying over the imagesTs folder across (this will not change at all.)