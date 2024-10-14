import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, Spacingd, SpatialPadd, CenterSpatialCropd, BorderPadd, DivisiblePadd
import os
import argparse 
import shutil
import sys 
import json
import math 
######################## This code base will assume that the dataset is already structured and selected, and the only difference with the new dataset will be that the images are downsampled.

def resampling_func(input_image_path, input_label_path, resampling_func, resampling_params, initial_size, final_size):
    
    input_dict = {"image":input_image_path, "label":input_label_path}

    size_after_downsample = np.round(np.array(initial_size)/np.array(resampling_params))
    #padding_size = [j for i,j in enumerate(final_size) if size_after_downsample[i] < j]
    #cropping_size = [j for i,j in enumerate(final_size) if size_after_downsample[i] > j]

    # if np.all((np.array(size_after_downsample) - np.array(final_size))%2):
    #    border_padding = [(j - size_after_downsample[i])/2  if size_after_downsample[i] < final_size[i] else 0 for i,j in enumerate(final_size)]
    # else:
    #     diff = np.array(final_size) - size_after_downsample
    #     border_padding = [[math.floor(diff[i]/2), math.ceil(diff[i]/2)] if diff > 0 else [0,0] for i in range(len(diff))]
    #     border_padding = [element for innerList in border_padding for element in innerList]
    

    if resampling_func == "Spacing":
        transforms_list = [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False), 
            EnsureChannelFirstd(keys=("image", "label")), 
            Spacingd(keys=("image","label"), pixdim=resampling_params, mode=("bilinear", "nearest")),
            #SpatialPadd(keys=("image, label"), )
            DivisiblePadd(keys=("image", "label"), k = [i/2 for i in final_size]),
            CenterSpatialCropd(keys=("image","label"), roi_size=final_size)

        ]
        transforms_composed = Compose(transforms_list)

    elif resampling_func == "Resize":
        transforms_list = [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False), 
            EnsureChannelFirstd(keys=("image", "label")),
            Resized(keys=("image", "label"), spatial_size=(resampling_params, resampling_params), mode=("area", "nearest"))
        ]
        transforms_composed = Compose(transforms_list)

    return transforms_composed(input_dict)


    
def save_image(output_dict, output_path, dict_key):
    output_image = np.array(output_dict[dict_key][0])
    output_affine = output_dict[dict_key].meta["affine"]
    output_header = nib.Nifti1Header() 

    output_header.set_xyzt_units(xyz=2)
    output_header.set_slope_inter(None, inter=None)
    output_header.set_qform(output_affine, code="scanner")
    output_header.set_sform(output_affine, code="scanner")

    nifti_image = nib.Nifti1Image(output_image, output_affine, header=output_header)
    nib.save(nifti_image, output_path)
    


def parse_arguments():
    parser = argparse.ArgumentParser("Data Preprocessing Downsampling")
    parser.add_argument("--base_dir", default="/home/parhomesmaeili/Radiology Datasets")
    parser.add_argument("--studies", nargs=1, default="BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2")#['Task09_Spleen', 'Tr']) # help="Folder for the study and the subcategory (e.g Ts or Tr)")
    parser.add_argument("--resampling_function", default="Spacing")
    parser.add_argument("--resampling_parameters", default="[2.0, 2.0, 2.0]")
    parser.add_argument("--final_size", default="[96, 96, 96]")
    parser.add_argument("--intial_size", default="[240, 240, 155]")
    return parser 

if __name__=="__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    input_dir = os.path.join(args.base_dir, args.studies)
    target_dir = input_dir + '_resized' #+ f'_{args.resampling_function}_{args.resampling_parameters}' 
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir, exist_ok=True)

    #Saving the resizing configuration somewhere:
    with open(os.path.join(target_dir, 'resizing_config.txt'), 'w') as f:
        f.write(f'Resampling function is {args.resampling_function} and resampling parameters are {args.resampling_parameters}')
    for root, dirs, _ in os.walk(input_dir):
        print(root)
        #print(files)
        index_dataset_name = root.split('/').index(args.studies) + 1
        try: 
            folder_subpath = os.path.join(*root.split('/')[index_dataset_name:])

            os.makedirs(os.path.join(target_dir, folder_subpath))
            #os.makedirs(root, exist_ok=True)
        except:
            continue
    for file in os.listdir(input_dir):     
        if file.endswith('.json'):
            shutil.copy(os.path.join(input_dir, file), os.path.join(target_dir, file))
    
    training_list = [file for file in os.listdir(os.path.join(input_dir, 'imagesTr')) if file.endswith('.nii.gz')]
    test_list = [file for file in os.listdir(os.path.join(input_dir, 'imagesTs')) if file.endswith('.nii.gz')]
    training_list.sort()
    test_list.sort()

    resampling_params = json.loads(args.resampling_parameters)
    resampling_function = args.resampling_function 

    initial_size = json.loads(args.intial_size)
    final_size = json.loads(args.final_size)

    for file in training_list:
        image_path = os.path.join(input_dir, "imagesTr", file)
        label_path = os.path.join(input_dir, "imagesTr", "labels", "final", file)

        output_dict = resampling_func(image_path, label_path, resampling_function, resampling_params, initial_size, final_size)
        
        image_output_path = os.path.join(target_dir, "imagesTr", file)
        label_output_path = os.path.join(target_dir, "imagesTr", "labels", "final", file)

        save_image(output_dict, image_output_path, "image")
        save_image(output_dict, label_output_path, "label")

    for file in test_list:
        image_path = os.path.join(input_dir, "imagesTs", file)
        label_path = os.path.join(input_dir, "imagesTs", "labels", "original", file)

        output_dict = resampling_func(image_path, label_path, resampling_function, resampling_params, initial_size, final_size)

        image_output_path = os.path.join(target_dir, "imagesTs", file)
        label_output_path = os.path.join(target_dir, "imagesTs", file)

        save_image(output_dict, image_output_path, "image")
        save_image(output_dict, label_output_path, "label")


    

