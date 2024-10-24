############################################
# NOTE: This code assumes that the dataset is in the Medical Decathlon format
######################################################################

import os
import sys
sys.path.append(os.path.join(os. path. expanduser('~'), 'MonaiLabel Deepedit++ Development', 'MONAILabelDeepEditPlusPlus'))
from monai.transforms import (
    LoadImaged,
    Compose, 
    Orientationd,
    EnsureChannelFirstd, 
)
import argparse
import numpy as np
import shutil
import json
from _ctypes import PyObj_FromPtr
import re
from monailabel.transform.writer import Writer 
from monailabel.transform.post import Restored 
from monai.transforms.transform import MapTransform, Randomizable, Transform
from collections.abc import Hashable, Mapping, Sequence, Sized
from monai.config import KeysCollection
from monai.data import MetaTensor

class ExtractChannel(MapTransform):
    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False, extracted_channels:list = None
    ):
        """
        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)
        self.extracted_channel = extracted_channels

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            original_mtensor = d[key]
            # print(original_mtensor.meta)
            # print(original_mtensor.shape)
            extracted_mtensor = original_mtensor[self.extracted_channel]
            
            del d[key]

            if len(extracted_mtensor.shape) == 4 and extracted_mtensor.shape[0] == 1:
                #d[key] = extracted_mtensor.squeeze()
                d[key] = extracted_mtensor #TODO: undo this if we do not use Restored

            else:
                d[key] = extracted_mtensor
        return d
    



class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr









def label_saving(result, output_dir, filename):
    #Saving the labels: 
    label = result[0]
    #test_dir = os.path.join(output_dir , "labels", "final")
    os.makedirs(output_dir, exist_ok=True)

    label_file = os.path.join(output_dir, filename)
    shutil.move(label, label_file)

    # print(label_json)
    # print(f"++++ Image File: {image_name_path}")
    # print(f"++++ Label File: {label_file}")
    
    return label_file







def json_generator(d):
    #Copy over json.dataset from old folder to the new one:
    print(d["input_base_dir"])
    shutil.copy(os.path.join(d["input_base_dir"], "dataset.json"), os.path.join(d["output_base_dir"], "dataset.json"))

    with open(os.path.join(d["output_base_dir"], "dataset.json"),'r') as f:
        json_dict = json.load(f)
    

    json_dict["numTraining"] = d["n_train"]
    json_dict["numTest"] = d["n_test"]

    train_pair_list = []
    test_pair_list = []

    for training_ext in d["training_list"]:
        sub_dict = dict()
        sub_dict["image"] = "./imagesTr/" + training_ext
        sub_dict["label"] = "./imagesTr/labels/final/" + training_ext 

        train_pair_list.append(sub_dict)

    for test_ext in d["test_list"]:
        sub_dict = dict()
        sub_dict["image"] = "./imagesTs/" + test_ext
        sub_dict["label"] = "./imagesTs/labels/original/" + test_ext

        test_pair_list.append(sub_dict)  
    
    json_dict["training"] = NoIndent(train_pair_list)
    json_dict["test"] = NoIndent(test_pair_list)

    if d["actions_config"][2] == "All":
        pass
    else:
        delete_channels = []
        for channel in json_dict["modality"].keys():
            if channel not in d["actions_config"][2]:
                delete_channels.append(channel)
    
        for channel in delete_channels:
            del json_dict["modality"][channel]
        
        tmp_image_size = json_dict["tensorImageSize"]
        new_dim = int(tmp_image_size[0]) - 1
        json_dict["tensorImageSize"] = str(new_dim) + "D"
    
    json_string = json.dumps(json_dict, cls=MyEncoder, indent=1)
    with open(os.path.join(d["output_base_dir"], "dataset.json"),'w') as f:
        f.write(json_string)

    return 

def image_conversion(d, train_list, test_list):
    #output_training_dir, output_training_labels_dir, output_test_dir, output_test_labels_dir 
    
    output_training_dir = d["output_training_dir"]
    #output_training_labels_dir = d["output_training_labels_dir"]

    output_test_dir = d["output_testing_dir"]
    #output_test_labels_dir = d["output_testing_labels_dir"]

    input_images_dir = d["input_images_dir"]
    #input_labels_dir = d["input_labels_dir"]

    extracted_channels = [int(i) for i in d["actions_config"][2]]
    
    transforms_list = [
        LoadImaged(keys=('image', 'default'), reader="ITKReader", image_only=False),#reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=('image', 'default')),
        Orientationd(keys=('image'), axcodes='RAS'),
        # #Orientationd(keys=('default'), axcodes="LAS"),
        ExtractChannel(keys=('image', 'default'), extracted_channels= extracted_channels),
        Restored(keys=('image'), ref_image = 'image', invert_orient=True)
    ]

    transform_comp = Compose(transforms=transforms_list, map_items = False)


    for filename in train_list:
        path_image = os.path.join(input_images_dir, filename)
        #path_label = os.path.join(input_labels_dir, filename)

        data = dict()
        data["image"] = path_image
        data["image_path"] = path_image
        data["default"] = path_image #TODO REMOVE THIS: THIS IS FOR DEBUGGING ORIENTATIONS
        #data["label"] = path_label

        data = transform_comp(data)
        #print(f'For image {filename} the original orientation array is \n {data["image_meta_dict"]["original_affine"]}')
 
        # meta_tmp = data.get("image_meta_dict").copy()
        
        # del data["image_meta_dict"]
        

        # meta = dict()
        # data["image_meta_dict"] = meta
        # meta["affine"] = meta_tmp.get("original_affine")

        result = Writer(label='image', nibabel=False)(data) #TODO: undo the nibabel writer use.

        label_saving(result, output_training_dir, filename)
     #label parameter for writer is the key for the image being saved (from a dict)
        #result = Writer(label='label')(data)
        #label_saving(result, output_training_labels_dir, filename)

    for filename in test_list:
        path_image = os.path.join(input_images_dir, filename)
        #path_label = os.path.join(input_labels_dir, filename)

        data = dict()
        data["image"] = path_image
        data["image_path"] = path_image
        data["default"] = path_image
        #data["label"] = path_label 

        data = transform_comp(data)



        # meta_tmp = data.get("image_meta_dict").copy()
        
        # del data["image_meta_dict"]
        

        # meta = dict()
        # data["image_meta_dict"] = meta
        # meta["affine"] = meta_tmp.get("original_affine")
            

        result = Writer(label='image', nibabel=False)(data)

        label_saving(result, output_test_dir, filename)
     #label parameter for writer is the key for the image being saved (from a dict)
        #result = Writer(label='label')(data)
        #label_saving(result, output_test_labels_dir, filename)
    

def dataset_reformatting(d, training_list, test_list):
    #This function takes the list of train/test images, in addition to the instructions on the channel extraction, to reformat and produce the new dataset.

    #If all channels being used, then just split according to the split percentage configuration:
    if d['actions_config'][2] == "All":

        #copy the images and labels for the training list 
        for filename in training_list:
            shutil.copy(os.path.join(d["input_images_dir"], filename), os.path.join(d["output_training_dir"], filename))
            shutil.copy(os.path.join(d["input_labels_dir"], filename), os.path.join(d["output_training_labels_dir"], filename))
        for filename in test_list:
        #copy the images and labels for the test list
            shutil.copy(os.path.join(d["input_images_dir"], filename), os.path.join(d["output_testing_dir"], filename))
            shutil.copy(os.path.join(d["input_labels_dir"], filename), os.path.join(d["output_testing_labels_dir"], filename))

    else:
        image_conversion(d, training_list, test_list)
        #This extracts the desired channels from the images for both sets of lists. The labels are single channel already

        #copy the images and labels for the training list 
        for filename in training_list:
            shutil.copy(os.path.join(d["input_labels_dir"], filename), os.path.join(d["output_training_labels_dir"], filename))
        for filename in test_list:
        #copy the images and labels for the test list
            shutil.copy(os.path.join(d["input_labels_dir"], filename), os.path.join(d["output_testing_labels_dir"], filename))

    print(f'There are {len(training_list)} images in the training set and {len(test_list)} images in the test set')
    return d 

def splitting_dataset_list(d):
    #This function generates the lists of the images which are to be distributed to training sets and test sets.

    #List the files:
    files = []
    for file in os.listdir(d["input_images_dir"]):
        if file.endswith(".nii.gz") or file.endswith(".nrrd"):
            files.append(file)
    files = np.array(files) 
   
    n_train = round(float(d["actions_config"][1]) * len(files))

    d["n_train"] = n_train 
    d["n_test"] = len(files) - n_train 
    #Shuffling the files:

    np.random.shuffle(files)

    #Splitting the files according to the split quantities. 

    train_list, test_list = np.split(files, [n_train])

    return train_list, test_list 

def dir_generator(d):
    input_base_dir = os.path.join(d["base_dir"], d["studies"][0])
    
    d["input_base_dir"] = input_base_dir
    
    d["input_images_dir"] = os.path.join(input_base_dir, "images" + d["studies"][1])
    d["input_labels_dir"] = os.path.join(input_base_dir, "labels" + d["studies"][1])

    output_base_dir = os.path.join(d["base_dir"], d["studies"][0] + '_' + f"Split_{d['actions_config'][0]}_proportion_{d['actions_config'][1]}_channels_{d['actions_config'][2]}")

    output_training_dir = os.path.join(output_base_dir, "imagesTr")
    output_testing_dir = os.path.join(output_base_dir, "imagesTs")

    output_training_labels_dir = os.path.join(output_base_dir, "imagesTr", "labels", "final")
    output_testing_labels_dir = os.path.join(output_base_dir, "imagesTs", "labels", "original")


    if os.path.isdir(output_base_dir):
        shutil.rmtree(output_base_dir)
    else:
        os.makedirs(output_base_dir)

    # if os.path.isdir(output_training_dir):
    #     shutil.rmtree(output_training_dir)
    # else:
    os.makedirs(output_training_dir)
    
    # if os.path.isdir(output_testing_dir):
    #     shutil.rmtree(output_testing_dir)
    # else:
    os.makedirs(output_testing_dir)
    
    # if os.path.isdir(output_training_labels_dir):
    #     pass
    # else:
    os.makedirs(output_training_labels_dir)
    
    # if os.path.isdir(output_testing_labels_dir):
    #     pass
    # else:
    os.makedirs(output_testing_labels_dir)

    
    d["output_base_dir"] = output_base_dir
    
    d["output_training_dir"] = output_training_dir
    d["output_testing_dir"] = output_testing_dir

    d["output_training_labels_dir"] = output_training_labels_dir
    d["output_testing_labels_dir"] = output_testing_labels_dir

    return d

def dataset_generator(config_dict):
    #print(config_dict)
    d = config_dict
    
    #Directory related code:

    d = dir_generator(d)

    #Generating the list of images used for training and testing to be extracted.
    training_list, test_list = splitting_dataset_list(d)

    dataset_reformatting(d, training_list, test_list)

    d["training_list"] = training_list
    d["test_list"] = test_list 
    return d 


def parse_arguments():
    parser = argparse.ArgumentParser("Data Preprocessing")
    parser.add_argument("--base_dir", default="/home/parhomesmaeili/Radiology Datasets")
    parser.add_argument("--studies", nargs=2, default=['Task09_Spleen','Tr'])#['Task01_BrainTumour', 'Tr'])#['Task09_Spleen', 'Tr']) # help="Folder for the study and the subcategory (e.g Ts or Tr)")
    parser.add_argument("--dataset_split", nargs="+", default=['True'], help="1st Entry: Bool for Split_Dataset")
    parser.add_argument("--dataset_split_percentages", default=['0.8'], nargs="+", help="Percentages btwn train-test split for each generated dataset desired, if blank then ")
    parser.add_argument("--channels_extracted", nargs="+", default=['0'])#['3'], help="List of channel combinations which are desired in the outputted generated datasets, e.g. [[1,2,3], [1,3,4], [1,2]]")
    return parser

def replaceitem(x, default):
    if x == '-':
        return default 
    else:
        return x 

if __name__ == "__main__":
    # export PYTHONPATH='/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/monailabel/':$PYTHONPATH

    parser = parse_arguments()
    args = parser.parse_args()
    
    configuration_dict = dict()

    configuration_dict["base_dir"] = args.base_dir
    configuration_dict["studies"] = args.studies 
    
    dataset_split = args.dataset_split
    dataset_split_percentages = args.dataset_split_percentages
    channels_extracted = args.channels_extracted

    #Default values for the ones which do not have entries.

    # if 
    # dataset_split_default = "True"
    # dataset_split_percentages_default = "0.75"
    # channels_extracted_default = "All"

    dataset_split = list(map(replaceitem, dataset_split, [dataset_split] * len(dataset_split)))
    dataset_split_percentages = list(map(replaceitem, dataset_split_percentages, [dataset_split_percentages] * len(dataset_split_percentages)))
    channels_extracted = list(map(replaceitem, channels_extracted, [channels_extracted] * len(channels_extracted)))


    #Code which creates the dataset configurations:


    for index in range(len(dataset_split)):
        configuration_dict["actions_config"] = [dataset_split[index], dataset_split_percentages[index], channels_extracted[index]]
        dataset_generator(configuration_dict)
        #Here we copy over the "somewhat defunct" dataset json description and make some alterations.
        json_generator(configuration_dict)


# Very generic Current Directory Layout MSD:

# -studies_folder
#     -imagesTr
#         |________labels
#                     |________final

#                     |________original
#     -imagesTs
#     -labelsTr
#     -dataset.json
    

#Desired Output Layout example:

# -studies_folder
#   |______  labels
#               |_______final
#                         |_______image_1.nii.gz
#                         |__________.....
#                         |_______image_30.nii.gz          
# : 
# json file
# image_1.nii.gz
# image_2.nii.gz
# ...
# 
# 
# image_30.nii.gz     
# 
# dataset.json (split of the dataset) 

    #TODO: write the function which splits the original dataset:

        #Probably do this by creating a list of the images for each subgroup

    #TODO: write the function which extracts the desired channels.



#There are four permutations (since the split percentage is just a "hyperparameter" for the )

#If split dataset and retain all channels:

    

    