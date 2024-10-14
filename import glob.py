# import glob
# import nibabel as nib
# import nrrd


# if __name__ == "__main__":
#     print('blahhh')
#     # Get a list of .nrrd files in a directory
#     nrrd_files = glob.glob('/home/parhomesmaeili/Desktop/OriginalDeepEditDummyOutput.nrrd')
#     print(nrrd_files)
#     # Loop through each .nrrd file
#     for nrrd_file in nrrd_files:
#         # Read the .nrrd file
#         data, header = nrrd.read(nrrd_file)
        
#         # Create a NIfTI1Image object
#         nifti_img = nib.Nifti1Image(data, affine=None)
        
#         # Update the NIfTI header with necessary information
#         nifti_img.header.set_data_dtype(data.dtype)
#         nifti_img.header.set_zooms(header['space directions'])
        
#         # Generate the output .nii file path by replacing the extension
#         nii_file = nrrd_file.replace('.nrrd', '.nii')
        
#         # Save the NIfTI1Image object as .nii file
#         nib.save(nifti_img, nii_file)


import SimpleITK as sitk

if __name__ == "__main__":
    img = sitk.ReadImage("/home/parhomesmaeili/Desktop/OriginalDeepEditDummyOutput.nrrd")
    sitk.WriteImage(img, "/home/parhomesmaeili/Desktop/OriginalDeepEditDummyOutput.nii.gz")