import subprocess
import os 
if __name__=="__main__":
    original_dataset_name = "/home/parhomesmaeili/Radiology Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2"
    dataset_name = "/home/parhomesmaeili/Radiology Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized"
    subprocess.call(['mkdir', dataset_name])
    subprocess.call(['mkdir', os.path.join(dataset_name, "imagesTr")])
    subprocess.call(['mkdir', os.path.join(dataset_name, "imagesTs")])
    subprocess.call(['mkdir', '-p', os.path.join(dataset_name, "imagesTr/labels/final")])
    subprocess.call(['mkdir', '-p', os.path.join(dataset_name, "imagesTs/labels/original")])
    
    #Processing the images:


    # for i in os.listdir(os.path.join(original_dataset_name, "imagesTr/labels/final/")):
    #     if i.endswith('.nii.gz'):

    #         print(i)
    #         #subprocess.call([''])
    #command_string = f'for i in  {os.path.join(original_dataset_name, "imagesTr/labels/final/")}*.nii.gz; do fslmaths $i -subsamp2  {os.path.join(dataset_name, "imagesTr/labels/final/")}`basename $i` -odt char; echo $i; done'
    #subprocess.Popen(['for', 'i','in', f'{os.path.join(original_dataset_name, "imagesTr/labels/final/*.nii.gz;")}', 'do', 'fslmaths', '$i', '-subsamp2', f'{os.path.join(dataset_name, "imagesTr/labels/final/")}`basename', '$i`', '-odt', 'char;', 'echo', '$i;', 'done'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(f'bash for i in  {os.path.join(original_dataset_name, "imagesTr/labels/final/")}*.nii.gz; do fslmaths $i -subsamp2  {os.path.join(dataset_name, "imagesTr/labels/final/")}`basename $i` -odt char; echo $i; done', shell=True, check=True, capture_output=True)
    #labels 'for i in  ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2/imagesTr/labels/final/*.nii.gz; do fslmaths $i -subsamp2  imagesTr/labels/final/`basename $i` -odt char; echo $i; done'
    #images 'for i in  ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2/imagesTr/labels/final/*.nii.gz; do fslmaths $i -subsamp2  imagesTr/labels/final/`basename $i`; echo $i; done'




    #PROBLEM: The labels are interpolated and therefore we end up with segmentation labels that have voxel values which are not in the original configuration:
    #instead we made our own FLIRT commands:

    #images for i in ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2/imagesTr/*.nii.gz; do flirt -in $i -ref $i -out ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized/imagesTr/`basename $i` -applyisoxfm 2 -datatype int -interp trilinear; echo $i; done
    #labels for i in ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2/imagesTs/labels/original/*.nii.gz; do flirt -in $i -ref $i -out ~/Radiology_Datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized/imagesTs/labels/original/`basename $i` -applyisoxfm 2 -datatype int -interp nearestneighbour; echo $i; done

