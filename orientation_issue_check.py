from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd)
import numpy as np
import torch 

if __name__ == '__main__':
    composed_transform = [LoadImaged(keys=('image', 'default'), image_only=False, reader="ITKReader"),
                          EnsureChannelFirstd(keys=('image', 'default')),]
    data = dict()

    path_image_new = '/home/parhomesmaeili/Orientation Folder Default Code/BRATS_316_processed.nii.gz'
    path_default = '/home/parhomesmaeili/Orientation Folder Default Code/BRATS_316.nii.gz'

    data["image"] = path_image_new
    data["image_path"] = path_image_new

    data["default"] = path_default
    data["default_path"] = path_default

    composed_transformations = Compose(transforms=composed_transform, map_items = False)
    output_data = composed_transformations(data)
    
    print(f'List of processed meta_dict keys {list(output_data["image_meta_dict"].keys())} \n')
    print(f'List of original meta dict keys {list(output_data["default_meta_dict"].keys())} \n')
    
    print('Processed BRATS Meta_Dict')
    print('\n')
    print(output_data["image_meta_dict"])
    print('\n')

    print('Default BRATS Meta_Dict')
    print('\n')
    print(output_data["default_meta_dict"])
    print('\n')

    for key in output_data["image_meta_dict"].keys():
        if not isinstance(output_data["image_meta_dict"][key], np.ndarray) and not isinstance(output_data['image_meta_dict'][key], torch.Tensor):
            if output_data["image_meta_dict"][key] != output_data["default_meta_dict"][key]:
                print(f'Difference in key {key} \n')
                print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                print(f'Original version is {output_data["default_meta_dict"][key]} \n')
        else:
            if output_data["image_meta_dict"][key].shape == output_data["default_meta_dict"][key].shape:
                if not np.array_equal(np.array(output_data["image_meta_dict"][key]), np.array(output_data["default_meta_dict"][key])):
                    print(f'Difference in key {key} \n')
                    print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                    print(f'Original version is {output_data["default_meta_dict"][key]} \n')
            else:
                    print(f'Difference in key {key} \n')
                    print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                    print(f'Original version is {output_data["default_meta_dict"][key]} \n')
        
    data = dict()

    path_image_new = '/home/parhomesmaeili/Orientation Folder Default Code/spleen_24_processed.nii.gz'
    path_default = '/home/parhomesmaeili/Orientation Folder Default Code/spleen_24.nii.gz'

    data["image"] = path_image_new
    data["image_path"] = path_image_new

    data["default"] = path_default
    data["default_path"] = path_default

    composed_transformations = Compose(transforms=composed_transform, map_items = False)
    output_data = composed_transformations(data)
    

    print(f'List of processed meta_dict keys {list(output_data["image_meta_dict"].keys())} \n')
    print(f'List of original meta dict keys {list(output_data["default_meta_dict"].keys())} \n')

    print('Processed Spleen Meta_Dict')
    print('\n')
    print(output_data["image_meta_dict"])
    print('\n')
    print('Default Spleen Meta_Dict')
    print('\n')
    print(output_data["default_meta_dict"])
    print('\n')
    for key in output_data["image_meta_dict"].keys():
        if not isinstance(output_data["image_meta_dict"][key], np.ndarray) and not isinstance(output_data['image_meta_dict'][key], torch.Tensor):
            if output_data["image_meta_dict"][key] != output_data["default_meta_dict"][key]:
                print(f'Difference in key {key} \n')
                print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                print(f'Original version is {output_data["default_meta_dict"][key]} \n')
        else:
            if output_data["image_meta_dict"][key].shape == output_data["default_meta_dict"][key].shape:
                if not np.array_equal(np.array(output_data["image_meta_dict"][key]), np.array(output_data["default_meta_dict"][key])):
                    print(f'Difference in key {key} \n')
                    print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                    print(f'Original version is {output_data["default_meta_dict"][key]} \n')
            else:
                    print(f'Difference in key {key} \n')
                    print(f'Processed version is \n {output_data["image_meta_dict"][key]} \n')
                    print(f'Original version is {output_data["default_meta_dict"][key]} \n')
    print('')
