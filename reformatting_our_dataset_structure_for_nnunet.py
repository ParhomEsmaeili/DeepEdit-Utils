import os
import shutil
import argparse 
import json
import sys 
sys.path.append('/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_1_2_3_4')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path_directory", default="/home/parhomesmaeili/Radiology_Datasets/ALL NNU-NET FOLDERS/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_nnU-net_1_2_3_4")
    args = parser.parse_args()

    imagesTr_path = os.path.join(args.dataset_path_directory, 'imagesTr')
    imagesTs_path = os.path.join(args.dataset_path_directory, 'imagesTs')

    try: 
        for filename in os.listdir(imagesTr_path):
            print(filename)
            head = filename[:-7]
            tail = filename[-7:]

            new_head = head + '_0000'
            new_string = new_head + tail

            os.rename(os.path.join(imagesTr_path, filename), os.path.join(imagesTr_path, new_string))
    except:
        pass 

    try: 
        for filename in os.listdir(imagesTs_path):
            print(filename)
            head = filename[:-7]
            tail = filename[-7:]

            new_head = head + '_0000'
            new_string = new_head + tail

            os.rename(os.path.join(imagesTs_path, filename), os.path.join(imagesTs_path, new_string))
    except:
        pass 