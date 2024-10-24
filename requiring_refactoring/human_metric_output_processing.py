import os

import argparse 
from os.path import dirname as up
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
import csv 
import json 
import sys
import shutil 
from pathlib import Path 
import copy 
import torch 

file_dir = os.path.join(os.path.expanduser('~'), 'MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus')
sys.path.append(file_dir)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class score_tool():
    def __init__(self, label_names, original_dataset_labels, label_mapping, human_measure, metric_type, metric_base, roi_scale_size):
    
        from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose
            )
        from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd  
        # from monai.metrics import Helper
        # from monai.metrics import Metric
        from monai.utils import MetricReduction
        # import torch

        self.original_dataset_labels = original_dataset_labels
        self.label_names = label_names
        self.label_mapping = label_mapping
        self.human_measure = human_measure 
        self.metric_type = metric_type 
        self.metric_base = metric_base 
        self.roi_scale_size = roi_scale_size
        # self.image_size = image_size 


        #Here we denote some pre-transforms for the  score computations. The same transforms list is not used because for LOCALITY the GT needs to be mapped 
        #to the same labels as the segmentation. For temporal consistency it is also required because the guidance points are stored using the labels being mapped to
        #as the keys in the dictionary storing them, but this is provided by self.label_names.

        if self.human_measure == "locality":
            self.transforms_list = [
            LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred", "gt")),
            Orientationd(keys=("pred", "gt"), axcodes="RAS"),  
            MappingLabelsInDatasetd(keys="gt", original_label_names=self.original_dataset_labels, label_names = self.label_names, label_mapping=self.label_mapping)
            ]

        elif self.human_measure == "temporal_consistency":
            self.transforms_list = [
            LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred", "gt")),
            Orientationd(keys=("pred", "gt"), axcodes="RAS"),  
            ]

        self.transforms_composition = Compose(self.transforms_list, map_items = False)

        # self._computation_class = Helper(  # type: ignore
        #         include_background= False,
        #         sigmoid = False,
        #         softmax = False, 
        #         activate = False,
        #         get_not_nans = False,
        #         reduction = MetricReduction.NONE, #MetricReduction.MEAN,
        #         ignore_empty = True,
        #         num_classes = None
        # )
        #self._computation_class = Metric()


        #Defining a class with the scoring utils, which can incorporate masks when computing scores. 
        self.scoring_util = ScoreUtils(metric_base)


    def __call__(self, pred_folder_path, gt_folder_path, image_name, guidance_points_set):

        #Here the guidance points set is a dictionary for the current segmentation under consideration. This dictionary contains the list of points for eachhhhhhh
        #class for the current iteration of segmentation. 

        
        pred_image_path = os.path.join(pred_folder_path, image_name)
        gt_image_path = os.path.join(gt_folder_path, image_name)

        input_dict = {"pred":pred_image_path, "gt":gt_image_path}
        output_dict = self.transforms_composition(input_dict)

        #We extract the image_size dimensions here because we want to implement the boxing of the guidance points in the RAS orientation. Therefore we need
        #to obtain the image size in that orientation so that we can have the appropriate box dimensions also.
        
        self.image_size = output_dict["pred"].shape[1:]

        image_mask = torch.Tensor(self.mask_generator(guidance_points_set))

        #We need to split the pred and GT into a class-by-class tensor.

        #Here we preapply our mask to each one-hot class segmentation. Compute the  score, and then compute the mean across them all. This means that 
        #we do not have to alter the original code for the class we are using. We also cannot apply the mask to the non-split segmentation since it would
        #treat the background and the NON-ROI region as the same thing.

        # one_hot_scores = dict()
        ignore_empty = True
        # for label_name, label_val in self.label_names.items():
        #     one_hot_seg = torch.where(output_dict["pred"] == label_val, 1, 0)
        #     one_hot_gt = torch.where(output_dict["gt"] == label_val, 1, 0)

        #     # one_hot_scores[label_name] = self._computation_class.compute_channel((one_hot_seg[0] * image_mask).bool(), (one_hot_gt[0] * image_mask).bool())
        #     one_hot_scores[label_name] = self.compute_channel((one_hot_seg[0] * image_mask).bool(), (one_hot_gt[0] * image_mask).bool(), ignore_empty)

        # if ignore_empty:
        #     #There will be some NaNs in instances where the GT label is empty. In these cases, we will just ignore these in the computation of the mean_,
        #     #using ignore_empty = True, instead of the alternative which would pad the  scores with 1s (in instances where there is no )
            
        #     summed_score = 0
        #     num = 0
        #     for label_names, one_hot_scores_val in one_hot_scores.items():
        #         if torch.isnan(one_hot_scores_val):
        #             print(f'NaN value for class {label_names}, image: {image_name}')
        #         else:
        #             summed_score += float(one_hot_scores_val)
        #             num += 1
        #     # output_score = sum([float(one_hot_scores[label_name]) for label_name in one_hot_scores.keys()])/len(one_hot_scores.keys())
        #     output_score = summed_score/num 
        # else:
        #     #In this case there will be no NaNs. We can just compute a mean.
        #     output_score = sum([float(one_hot_scores[label_name]) for label_name in one_hot_scores.keys()])/len(one_hot_scores.keys())

        
        output_score = self.scoring_util(ignore_empty, image_mask, pred=output_dict["pred"], gt=output_dict["gt"])
        
        return output_score

    def mask_generator(self, guidance_points_set):
        #This takes the dictionary which contains the guidance points for each class, and generates the set of masks for each class
        
        if self.metric_type == "ellipsoid":

            roi_size = np.round(np.array(self.image_size) * self.roi_scale_size) 
            mask = self.roi_mask_apply(guidance_points_set, roi_size, ellipsoid_shape_bool=True)
        #We round in the case that we have a decimal dimension size. The roi_size dimensions are also for only one quadrant size (i.e. half in each direction)

        elif self.metric_type == "cuboid":

            roi_size = np.round(np.array(self.image_size) * self.roi_scale_size)
            mask = self.roi_mask_apply(guidance_points_set, roi_size, ellipsoid_shape_bool=False)
        
        elif self.metric_type == "distance":

            mask = self.distance_mask(guidance_points_set)
        
        elif self.metric_type == "2d_intersections":

            mask = self.intersection_mask(guidance_points_set)
        
        if self.human_measure == "locality":
            #If locality is the measure then no change required.
            #output_mask_current_class
            output_mask = mask 

        elif self.human_measure == "temporal_consistency":
            #If temporal consistency is the measure, then we need to invert the mask.
            #output_mask_current_class
            output_mask = 1 - mask

        # masks[label_class] = output_mask_current_class

        return output_mask
    
    def roi_mask_apply(self, guidance_points_set, roi_size, ellipsoid_shape_bool):
        #We will not be extracting separate masks for each class because the intention should be that whatever desired characteristic is being exhibited, should
        #be doing so across all classes. And even if 

        #Initialise a tensor extracting the regions around the guidance points.

        roi_mask_tensor = np.zeros(self.image_size)

        # masks = dict()
        for label_class, guidance_points in guidance_points_set.items():
            #guidance_points = guidan
            

            for guidance_point in guidance_points:
                #obtain the extreme points of the cuboid which will be assigned as the box region:
                min_maxes = []
                
                if ellipsoid_shape_bool:
                    pass 
                else: 
                    for index, coordinate in enumerate(guidance_point):
                        #For each coordinate, we obtain the extrema points.
                        dimension_min = int(max(0, coordinate - roi_size[index]))
                        dimension_max = int(min(self.image_size[index] - 1, coordinate + roi_size[index]))

                        min_max = [dimension_min, dimension_max] 
                        min_maxes.append(min_max)

                    if len(self.image_size) == 2:
                    #If image is 2D            
                        roi_mask_tensor[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1]] = 1
                    elif len(self.image_size) == 3:
                        #If image is 3D:
                        roi_mask_tensor[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1], min_maxes[2][0]:min_maxes[2][1]] = 1
                
        return roi_mask_tensor
    
    def compute_channel(self, y_pred: torch.Tensor, y: torch.Tensor, ignore_empty) -> torch.Tensor:
        """"""
        y_o = torch.sum(y)
        if y_o > 0:
            return (2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
        if ignore_empty:
            return torch.tensor(float("nan"), device=y_o.device)
        
        denorm = y_o + torch.sum(y_pred)

        #Should ignore_empty == True? Arguable that yes we should just ignore the empty GTs because for a multiclass segmentation, the non-existent GT
        #, in the case where there is still a prediction, would have that false-positive error reflected in another class anyways!

        #However, if we don't ignore it then probably:

        #If GT is empty and prediction is empty: We make a slight change to the code, and change this from being a 1 to being a nan. This is because it would
        #pad the  score which would already have been computed for a different class. Instead it should just not contribute!
        if denorm <= 0:
            return torch.tensor(float("nan"), device=y_o.device)
        #If GT is empty but prediction is not. In this case we should be punishing the fact that there is a mistaken prediction (false positive).
        return torch.tensor(0.0, device=y_o.device)


class ScoreUtils():
    def __init__(self, score_base):
        self.score_base = score_base 
        self.supported_bases = ["Dice", "Error Rate"]

        if self.score_base.title() not in self.supported_bases:
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected metric base is not supported")

    def dice_score(self, ignore_empty, image_mask, pred, gt):
        return 
    
    def error_rate(self, ignore_empty, image_mask, pred, gt):
        return 

    def __call__(self, ignore_empty, image_mask, pred, gt):
        
        if self.score_base == "Dice":
            output_score = self.dice_score(ignore_empty, image_mask, pred, gt)
        
        elif self.score_base == "Error Rate":
            output_score = self.error_rate(ignore_empty, image_mask, pred, gt)


        return output_score 






def guidance_point_extraction(guidance_point_files_list, initialisation, framework):
    #If framework is deepedit (original) then we need to output a guidance point dictionary for both the original (which contains all guidance points) and
    #for the new points per iteration.

    if framework == "deepeditPlusPlus":
        #In this case we extract the guidance_points normally.
        guidance_points_dict = dict()

        for guidance_point_file in guidance_point_files_list:
            with open(guidance_point_file, 'r') as f:
                saved_dict = json.load(f)
                guidance_points_dict[Path(guidance_point_file).stem] = copy.deepcopy(saved_dict)

        guidances_per_iter_dict = None #copy.deepcopy(guidance_points_dict)
    
    elif framework == "deepedit":
        #In this case we extract both the guidance points normally, AND we also generate the "new guidance per iters" dictionary
        guidance_points_dict = dict()
        for guidance_point_file in guidance_point_files_list:
            with open(guidance_point_file, 'r') as f:
                saved_dict = json.load(f)
                guidance_points_dict[Path(guidance_point_file).stem] = copy.deepcopy(saved_dict)
        
        # Computing the new iters for all of the iterations with guidance points as illustrated by the guidance point files list.

        guidances_per_iter_dict = dict() 
        guidances_per_iter_dict[Path(guidance_point_files_list[0]).stem] = copy.deepcopy(guidance_points_dict[Path(guidance_point_files_list[0]).stem])


        for j, guidance_point_file in enumerate(guidance_point_files_list[1:]):

            current_guidance_dict = guidance_points_dict[Path(guidance_point_file).stem]
            previous_guidance_dict = guidance_points_dict[Path(guidance_point_files_list[j]).stem]

            #This dict will hold for all  the images, the dictionaries of the NEW guidance points across the classes for the current iteration
            new_guidance_dict = dict() 
            #For each image, find the difference in the guidance points between iterations
            for image in current_guidance_dict.keys():

                new_guidance_per_image = dict()
                #For each class label find the differencein the guidance points per iterations, for the given image.
                for class_label in current_guidance_dict[image].keys():
                    #Computing the new guidance points for each class label 
                    current_guidance_points = current_guidance_dict[image][class_label]
                    previous_guidance_points = previous_guidance_dict[image][class_label]


                    new_guidance_class_label = [point for point in current_guidance_points if point not in previous_guidance_points]

                    new_guidance_per_image[class_label] = new_guidance_class_label
                
                #Appending the new guidance points to the dict of "new guidances" per iteration.
                new_guidance_dict[image] = new_guidance_per_image
            
            #Adding the new guidance dict for the current iteration to the dictionary containing the "new guidances" across all iterations.
            guidances_per_iter_dict[Path(guidance_point_file).stem] = new_guidance_dict 
        

    output_guidance_dicts = dict()
    if guidance_points_dict is not None:
        output_guidance_dicts['original_guidance'] = guidance_points_dict #        [guidance_points_dict, guidances_per_iter_dict]
    if guidances_per_iter_dict is not None:
        output_guidance_dicts['guidance_per_iter'] = guidances_per_iter_dict 

    return output_guidance_dicts

def folder_list_generator(img_directory, segmentation_task, initialisation, human_measure):
    import re 

    if human_measure == "locality":
        #Obtaining list of image names, not done in numeric order.
        image_names = [x for x in os.listdir(img_directory) if x.endswith('.nii.gz')]
        #Obtaining the 
        gt_image_folder = os.path.join(img_directory, 'labels', 'original')
        final_image_folder = 'final'

        if segmentation_task == "deepedit":
        
            if initialisation == "autoseg":
                iteration_folders = [x for x in os.listdir(os.path.join(img_directory, 'labels')) if x.startswith('deepedit_iteration')]
                iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
                #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
                #this is needed because the iterations need to be in order.
                
                final_folder_list = iteration_folders + [final_image_folder]

            elif initialisation == "deepgrow":

                initialisation_folder = initialisation

                iteration_folders = [x for x in os.listdir(os.path.join(img_directory, 'labels')) if x.startswith('deepedit_iteration')]
                iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
                #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
                #this is needed because the iterations need to be in order.
                
                final_folder_list = [initialisation_folder] + iteration_folders + [final_image_folder]

        
        elif segmentation_task == "deepgrow":

                final_folder_list = final_image_folder




    #For temporal consistency measurement scores, this onlyyyyyyy applies to methods with iterative refinement. NOT for methods with only one iteration.

    elif human_measure == "temporal_consistency":
        #Obtaining list of image names, not done in numeric order.
        image_names = [x for x in os.listdir(img_directory) if x.endswith('.nii.gz')]
        #Obtaining the 
        gt_image_folder = None #os.path.join(img_directory, 'labels', 'original')
        final_image_folder = 'final'

        if segmentation_task == "deepedit":
        
            # if initialisation == "autoseg":
            #     iteration_folders = [x for x in os.listdir(os.path.join(img_directory, 'labels')) if x.startswith('deepedit_iteration')]
            #     iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #     #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #     #this is needed because the iterations need to be in order.
                
            #     final_folder_list = iteration_folders + final_image_folder
            
            # elif initialisation == "deepgrow":
                
            initialisation_folder = initialisation

            iteration_folders = [x for x in os.listdir(os.path.join(img_directory, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.
            
            final_folder_list = [initialisation_folder] + iteration_folders + [final_image_folder]
        
        # elif segmentation_task == "deepgrow":
            



    folder_set_dict = dict()
    
    folder_set_dict["image_names"] = image_names
    folder_set_dict["gt_image_folder"] = gt_image_folder
    folder_set_dict["final_folder_list"] = final_folder_list

    return folder_set_dict

def score_computation(img_directory, inference_tasks, results_save_dir, jobs, study_name, human_measure, metric_type, metric_base, roi_scale_size):
    #This method should compute the  scores for the set of images and task which has been provided. It should 
    #save this in a csv file, the  scores returned should provide the  scores across all the sets of images
    #that have been provided. E.g. if we have a specific sub-task, and we train a model on a set of diff datasets and  
    #test that model on that specific sub-task, we would like it to save the  scores across all these sets. It gives
    #us a broader range of test outputs for which we can assess it on? 

    if "compute" in jobs:

        import re 

        if human_measure == "locality":
            '''
            For a locality based measure, we will be implementing a method which uses the guidance points to generate masks, which we apply to the Seg. and
            GT to obtain masked  scores.
            '''
            
            #Here we extract the guidance_points directory path which we will be using to guide the human-guided measures.
            guidance_points_dir = os.path.join(img_directory, 'labels', 'guidance_points')




            # #Here we will generate the paths for the images with which we want to compute  scores
            
            # #Obtaining list of image names, not done in numeric order:
            # image_names = [x for x in os.listdir(img_directory) if x.endswith('.nii.gz')]
            # gt_image_folder = os.path.join(img_directory, 'labels', 'original')
            # final_image_folder = os.path.join(img_directory,'labels', 'final')

            #Extracting the name of the framework under consideration.
            framework = inference_tasks[0]
            if framework == "deepeditplusplus":
                framework = "deepeditPlusPlus"
            
            #Extracting the name of the dataset under consideration
            dataset_name = study_name[:-9]

            #Extracting the label mapping path, so that we can map the GT labels as required.
            label_config_path = os.path.join(file_dir, "monailabel", framework, dataset_name + '_label_configs.txt')
            
            ################### Importing the label configs dictionary #####################

            with open(label_config_path) as f:
                config_dict = json.load(f)

            config_labels = config_dict["labels"]
            config_original_dataset_labels = config_dict["original_dataset_labels"]
            config_label_mapping = config_dict["label_mapping"]


            #Creating our output folders to save the results.
            if os.path.exists(results_save_dir) == True:
                shutil.rmtree(results_save_dir)
            os.makedirs(results_save_dir)


            #Initialisation of the  score computation tool.
            _computer = score_tool(config_labels, config_original_dataset_labels, config_label_mapping, human_measure, metric_type, metric_base, roi_scale_size)

            if inference_tasks[1] == "deepedit":

                #For deepedit: if it initialises with autoseg then the locality computation need only be implemented for the refinement iterations. 
                #If it initialises wiiiith deepgrow then the locality computation would be implemented for all iterations.

                #Extracting the name of the initialisation
                initialisation = inference_tasks[2]

                if initialisation == "autoseg":
                
        
                    folder_and_files_dict = folder_list_generator(img_directory, inference_tasks[1], initialisation, human_measure) #iteration_folders + final_image_folder 
                    
                    image_names = folder_and_files_dict["image_names"]
                    gt_image_folder = folder_and_files_dict["gt_image_folder"]
                    final_folder_list = folder_and_files_dict["final_folder_list"]
                    #Final folder list denotes the "final" folders which images are stored in. I.e. the deepest layer. 

                    #Here we obtain the paths for the json files that contain the guidance points for each iteration:

                    guidance_points_files = [os.path.join(guidance_points_dir, f'{folder}.json') for folder in final_folder_list[:-1]] + [os.path.join(guidance_points_dir, f'{final_folder_list[-1]}_iteration.json')]

                    #Creating a dict to store all of the guidance points for all images & iterations:

                    # guidance_points_dict = dict()

                    # for guidance_point_file in guidance_points_files:
                    #     with open(guidance_point_file, 'r') as f:
                    #         saved_dict = json.load(f)
                    #         guidance_points_dict[Path(guidance_point_file).name] = saved_dict

                    guidance_points_dicts = guidance_point_extraction(guidance_points_files, initialisation, framework)


                    for guidance_points_dict_key, guidance_points_dict in guidance_points_dicts.items():
                        #For each of the dict collections (may just be only the original, and in the case of deepedit it may ALSO be the "new iter only" one)
                        # guidance_points_dict = guidance_points_dic
                        for image in image_names:
                            scores = [image]
                            
                            image_name_only = image.split('.')[0]
                            for j, iteration_folder in enumerate(final_folder_list): #Adding the  scores on the intermediary iterations

                                #Extracting the guidance points for the current image in the current iteration
                                guidance_point_set = guidance_points_dict[Path(guidance_points_files[j]).stem][image_name_only]


                                scores.append(_computer(os.path.join(img_directory, 'labels', iteration_folder), gt_image_folder, image, guidance_point_set))

                            os.makedirs(os.path.join(results_save_dir, guidance_points_dict_key), exist_ok=True)
                            with open(os.path.join(results_save_dir, guidance_points_dict_key, 'score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores)

                elif initialisation == "deepgrow":
                    
    
                    folder_and_files_dict = folder_list_generator(img_directory, inference_tasks[1], initialisation, human_measure)
                    
                    image_names = folder_and_files_dict["image_names"]
                    gt_image_folder = folder_and_files_dict["gt_image_folder"]
                    final_folder_list = folder_and_files_dict["final_folder_list"]

                    #Here we obtain the paths for the json files that contain the guidance points for each iteration:

                    guidance_points_files = [os.path.join(guidance_points_dir, f'{folder}.json') for folder in final_folder_list[:-1]] + [os.path.join(guidance_points_dir, f'{final_folder_list[-1]}_iteration.json')]

                    #Creating a dict to store all of the guidance points for all images & iterations:

                    
                    guidance_points_dicts = guidance_point_extraction(guidance_points_files, initialisation, framework)
                    
                    # guidance_points_dict = dict()


                    # for guidance_point_file in guidance_points_files:
                    #     with open(guidance_point_file, 'r') as f:
                    #         saved_dict = json.load(f)
                    #         guidance_points_dict[Path(guidance_point_file).name] = saved_dict


                    for guidance_points_dict_key, guidance_points_dict in guidance_points_dicts.items(): 

                        for image in image_names:
                            scores = [image]
                            
                            image_name_only = image.split('.')[0]
                            for j, iteration_folder in enumerate(final_folder_list): #Adding the  scores on the intermediary iterations

                                #Extracting the guidance points for the current image in the current iteration
                                guidance_point_set = guidance_points_dict[Path(guidance_points_files[j]).stem][image_name_only]


                                scores.append(_computer(os.path.join(img_directory, 'labels', iteration_folder), gt_image_folder, image, guidance_point_set))

                            os.makedirs(os.path.join(results_save_dir, guidance_points_dict_key), exist_ok=True)
                            with open(os.path.join(results_save_dir, guidance_points_dict_key, 'score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores)
                        

            elif inference_tasks[1] == 'deepgrow':

                #In this case we are computing the output for the deepgrow-only segmentation.
                #gt_image_folder = os.path.join(img_directory, 'labels', 'original')

                folder_and_files_dict = folder_list_generator(img_directory, inference_tasks[1], None, human_measure)
                # guidance_points_dict = dict()
                image_names = folder_and_files_dict["image_names"]
                gt_image_folder = folder_and_files_dict["gt_image_folder"]
                final_image_folder = folder_and_files_dict["final_folder_list"]
                # final_image_folder = final_folder_list

                guidance_point_file = os.path.join(img_directory, 'labels', 'guidance_points', 'final_iteration.json')

                with open(guidance_point_file, 'r') as f:
                    saved_dict = json.load(f)
                    guidance_points_dict= saved_dict

                for image in image_names:
                    scores = [image]
                    
                    image_name_only = image.split('.')[0]
                    guidance_point_set = guidance_points_dict[image_name_only]

                    scores.append(_computer(os.path.join(img_directory, 'labels', final_image_folder), gt_image_folder, image, guidance_point_set))
                    #We only have one set of points to consider because it is only one iteration! 
                    os.makedirs(os.path.join(results_save_dir, 'original_guidance'), exist_ok=True)
                    with open(os.path.join(results_save_dir, 'original_guidance', 'score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores)
                # return scores 


        elif human_measure == "temporal_consistency":
            '''
            For a temporal consistency based measure, we will be implementing a method which uses the guidance points to generate masks, 
            which we apply to the Seg. and prior iterations to obtain masked  scores. In this situation the "GT" would be the previous segmentations
            '''

            #Here we extract the guidance_points directory path which we will be using to guide the human-guided measures.
            guidance_points_dir = os.path.join(img_directory, 'labels', 'guidance_points')

            #Here we will generate the paths for the images with which we want to compute  scores
            
            # #Obtaining list of image names, not done in numeric order:
            # image_names = [x for x in os.listdir(img_directory) if x.endswith('.nii.gz')]
            
            # # gt_image_folder = os.path.join(img_directory, 'labels', 'original')

            # final_image_folder = os.path.join(img_directory,'labels', 'final')

            framework = inference_tasks[0]
            if framework == "deepeditplusplus":
                framework = "deepeditPlusPlus"

            dataset_name = study_name[:-9]

            label_config_path = os.path.join(file_dir, "monailabel", framework, dataset_name + '_label_configs.txt')
            
            ################### Importing the label configs dictionary #####################

            with open(label_config_path) as f:
                config_dict = json.load(f)

            config_labels = config_dict["labels"]
            config_original_dataset_labels = config_dict["original_dataset_labels"]
            config_label_mapping = config_dict["label_mapping"]

            if os.path.exists(results_save_dir) == True:
                shutil.rmtree(results_save_dir)
            os.makedirs(results_save_dir)

            _computer = score_tool(config_labels, config_original_dataset_labels, config_label_mapping, human_measure, metric_type, metric_base, roi_scale_size)



            if inference_tasks[1] == "deepedit":
                
                
                #There is no functional difference between a deepgrow or autoseg initialisation outside of the paths, because we start from the first refinement
                #step anyways.

                initialisation = inference_tasks[2]

                folder_and_files_dict = folder_list_generator(img_directory, inference_tasks[1], initialisation, human_measure)
                image_names = folder_and_files_dict["image_names"]
                gt_image_folder = folder_and_files_dict["gt_image_folder"]
                final_folder_list = folder_and_files_dict["final_folder_list"]

                #Setting the iteration_folder variable. We use this to iterate through because to compute across iterations we would need to offset the
                #index by 1. We will iterate through the folders by iterating using index instead.
                iteration_folder = final_folder_list

                #TODO: Fill in the guidance extractions now. And the generation of the  scores ffor each (if it is the original deepedit configuration)
                #Here we list the guidance points files to extract the points from.
                


                #Here we split between the two different initialisations, because only one initialisation has guidance points associated with it. 

                if initialisation == "deepgrow":

                    guidance_points_files = [os.path.join(guidance_points_dir, f'{folder}.json') for folder in final_folder_list[:-1]] + [os.path.join(guidance_points_dir, f'{final_folder_list[-1]}_iteration.json')]

                        #Creating a dict to store all of the guidance points for all images & iterations:


                    guidance_points_dicts = guidance_point_extraction(guidance_points_files, initialisation, framework)

                    for guidance_points_dict_key, guidance_points_dict in guidance_points_dicts.items():
                        for image in image_names:
                            scores = [image]

                            image_name_only = image.split('.')[0]

                            #Here we will compute the consistency score across the iterations

                            #We need to extract the guidance points for the current instance
                            # for iteration, iteration_folder in enumerate(final_folder_list):
                            for iteration in range(len(final_folder_list) - 1): 

                                #Extracting the guidance points for the current image in the current iteration
                                guidance_point_set = guidance_points_dict[Path(guidance_points_files[iteration + 1]).stem][image_name_only] 

                                scores.append(_computer(os.path.join(img_directory, 'labels', iteration_folder[iteration + 1]), os.path.join(img_directory, 'labels', iteration_folder[iteration]), image, guidance_point_set))
                            
                            # create the final output save directory
                            os.makedirs(os.path.join(results_save_dir, guidance_points_dict_key), exist_ok=True)

                            with open(os.path.join(results_save_dir, guidance_points_dict_key, 'score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores)
                #we separate the two initialisations because deepgrow initialisation has an extra set of guidance points, which we need to account for when
                #computing our guidance per iter points. Furthermore, the fact that the deepgrow initialisation has an extra set of guidance points in addition to the 
                #refinement points, means that when computing temporal consistency at refinement iter 1 (requiring guidance points at iter 1) we would have N + 1 guidance sets
                #so we need to offset the index by 1 when extracting the guidance points.
                elif initialisation == "autoseg":
                    guidance_points_files = [os.path.join(guidance_points_dir, f'{folder}.json') for folder in final_folder_list[1:-1]] + [os.path.join(guidance_points_dir, f'{final_folder_list[-1]}_iteration.json')]

                        #Creating a dict to store all of the guidance points for all images & iterations:


                    guidance_points_dicts = guidance_point_extraction(guidance_points_files, initialisation, framework)

                    for guidance_points_dict_key, guidance_points_dict in guidance_points_dicts.items():
                        for image in image_names:
                            scores = [image]

                            image_name_only = image.split('.')[0]

                            #Here we will compute the consistency score across the iterations

                            #We need to extract the guidance points for the current instance
                            # for iteration, iteration_folder in enumerate(final_folder_list):
                            for iteration in range(len(final_folder_list) - 1): 

                                #Extracting the guidance points for the current image in the current iteration (it is not iteration + 1 because this is autoseg!)
                                guidance_point_set = guidance_points_dict[Path(guidance_points_files[iteration]).stem][image_name_only] 

                                scores.append(_computer(os.path.join(img_directory, 'labels', iteration_folder[iteration + 1]), os.path.join(img_directory, 'labels', iteration_folder[iteration]), image, guidance_point_set))
                            
                            os.makedirs(os.path.join(results_save_dir, guidance_points_dict_key), exist_ok=True)
                            with open(os.path.join(results_save_dir, guidance_points_dict_key, 'score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores)


        # if inference_tasks[1] == "deepgrow":
        #     #In this case, iterate through each guidance point dict name, and append to the bottom of the file.
        #     for guidance_points_dict_key in guidance_points_dicts.keys():
        #         #For each of the dict collections (may just be only the original, and in the case of deepedit it may ALSO be the "new iter only" one)
                
        #         with open(os.path.join(results_save_dir, guidance_points_dict_key, 'score_results.csv'),'a') as f:
        #             writer = csv.writer(f)
        #             writer.writerow(scores)

        # else:
        #     pass
        #     #In this case just go through the "original guidance" spreadsheet.

def score_extraction(results_save_dir):
    with open(os.path.join(results_save_dir, 'score_results.csv'), newline='') as f:
        score_reader = csv.reader(f, delimiter=' ', quotechar='|')
        first_row = f.readline()
        first_row = first_row.strip()
        #print(first_row)
        #n_cols = first_row.count(',') + 1 
        scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
        #print(scores)
        for row in score_reader:
            row_str_list = row[0].split(',')
            #print(row_str_list)
            for index, string in enumerate(row_str_list):
                if index > 0:
                    scores[index].append(float(string))
                elif index == 0:
                    scores[index].append(string)
    return scores 

def score_collection(results_save_dir, image_subtasks, score_files_base_dir, rejection_value): #, infer_runs):
    #obtaining the paths for all of the  score files we want to merge together:
    score_paths = [os.path.join(score_files_base_dir, image_subtask) for image_subtask in image_subtasks]

    #extracting the  scores and collecting them together.
    all_scores = []

    for _path in score_paths:
        with open(os.path.join(_path, 'score_results.csv'), newline='') as f:
            score_reader = csv.reader(f, delimiter=' ', quotechar='|')
            first_row = f.readline()
            first_row = first_row.strip()
            #print(first_row)
            #n_cols = first_row.count(',') + 1 
            scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
            #print(scores)
            for row in score_reader:
                row_str_list = row[0].split(',')
                #print(row_str_list)
                for index, string in enumerate(row_str_list):
                    if index > 0:
                        scores[index].append(float(string))
                    elif index == 0:
                        scores[index].append(string)

            all_scores.append(scores)
        
    final_output_scores = all_scores[0]

    for score_set in all_scores[1:]:
        #score_set_index = index + 1
        #current_score_set = all_scores[score_set_index]

        for output_index in range(len(final_output_scores)):
            
            final_output_scores[output_index] += score_set[output_index] #current_score_set[output_score_index]

    #Here we will compute the mean of all the columns so that we can view them when necessary, without needing to keep running the plotting script.
    #Also use can use this mean as part of the plotting function. 

    #Here we will compute the mean of all the columns so that we can view them when necessary, without needing to keep running the plotting script.
    #Also use can use this mean as part of the plotting function. 

    score_average = ['averages']
    non_rejected_rows = []
    for score_row_index in range(len(final_output_scores[0])): #final_output_scores[1:]:

        score_row = [final_output_scores[j][score_row_index] for j in range(len(final_output_scores))]
        if all(score_row[1:]) >= rejection_value:
        
        # accepted_scores = [i for i in score_column if i >= rejection_value]
            non_rejected_rows.append(score_row[1:])

        # score_average.append(sum(accepted_scores)/len(accepted_scores))
    totals = copy.deepcopy(non_rejected_rows[0])
    count = 1
    for non_rejected_row in non_rejected_rows[1:]:
        for index, val in enumerate(non_rejected_row):
            totals[index] += val 
        count += 1
    
    for total in totals:
        score_average.append(total/count)

    for i in range(len(score_average)):
        final_output_scores[i].append(score_average[i])

    #compute standard deviations. 
    stdevs = np.std(np.array(non_rejected_rows), axis=0)
    #appending them to the file

    final_output_scores[0].append('stdevs')
    for i in range(len(stdevs)):
        final_output_scores[i + 1].append(stdevs[i])


    # #Creating the save directory.   
    if os.path.exists(results_save_dir):
        shutil.rmtree(results_save_dir)

    os.makedirs(results_save_dir, exist_ok=True)

    with open(os.path.join(results_save_dir, 'score_results.csv'),'a') as f:
        writer = csv.writer(f)
        
        for i in range(len(final_output_scores[0])):
            output_row = [sublist[i] for sublist in final_output_scores]
            writer.writerow(output_row)

    return 

def _visualisation(scores, task_configs, results_dir, run_infer_string, guidance_set, rejection_value):
    import seaborn as sb
    import pandas as pd


    inference_task = task_configs[1]
    print(f'Inference task is {inference_task}')
    if inference_task == "deepedit":
        initialisation = task_configs[2]
        num_clicking_iters = task_configs[3] 
        task_title = f'{initialisation.capitalize()} initialisation, {num_clicking_iters} {inference_task.capitalize()} iterations'
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title(f'{task_title[:-1]}  scores')
        ax.set_xlabel('Clicking Iteration')
        ax.set_ylabel(' Score')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        task_title = inference_task.capitalize()
        ax.set_title(f'{inference_task.capitalize()}  Scores')
        ax.set_ylabel(' Score')
    
    # If we are doing a deepedit output, then we just want a moving average across the iterations (possibly with some range bars?)

    # If we are doing an autoseg or deepgrow output, then we just want a distribution of the scores. 

    ################ Removing the failure cases from the mean computation, and printing the list of failure cases into a separate text file to use as exemplars. ###################
    #print(np.array(scores).shape)
    tmp_array = np.array(scores)
    # print(tmp_array[0, :])
    scores_array = tmp_array[1:,:].astype(np.float64)
    scores_array = scores_array.T

    #We remove the average row because its not necessary.
    scores_array = scores_array[:, :]
    # print(scores_array[-1, :])
    image_names = tmp_array[0,:]
    # print(image_names)

    #Setting failure case  score:
    failure_ = rejection_value
    # print(scores_array.shape)
    failure_images = dict()
    row_removal_ins = []
    #we iterate up until shape - 2 because we do not want to examine the average or the stdev rows
    for index in range(scores_array.shape[0] - 2): #np.array(scores):
        sub_array = scores_array[index, :]
        #print(sub_array)
        if np.any(sub_array < failure_):
            
            # print(sub_array)
            # print(image_names[index])

            failure_images[image_names[index]] = sub_array.tolist()
            row_removal_ins.append(index)
    
    #scores = [i if i >= failure_ in scores]
    #scores = []
    ################################################################################################################################################################################

    ######################### For the initialisation failures/general failures, save the names of the images #######################################################

    with open(os.path.join(results_dir,'failure_cases.txt'), 'w') as f:
        f.write(json.dumps(failure_images))
    
    #Base directory which we will place the folder in for saving plots:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(results_dir))), 'Plots', "Standalone")
    folder_save_name = f"model_{datetime}_checkpoint_{checkpoint}/{run_infer_string}/{guidance_set}"


    save_dir = os.path.join(base_dir, folder_save_name)
    os.makedirs(save_dir, exist_ok=True)

    ###### Remove the failure case rows ##########
    # print(np.sum(scores_array, axis=0)/249)
    # print(scores_array[row_removal_ins,:][0][0])

    final_scores_array = np.delete(scores_array, row_removal_ins, axis=0)

    final_scores_array_scores = final_scores_array[:-2,:]
    scores_mean = final_scores_array[-2, :]
    scores_stdev = final_scores_array[-1, :]

    # print(final_scores_array.shape)
    ####################################################################################################################################

    if inference_task == "deepedit":
        x = np.array(range(final_scores_array_scores.shape[1]))
        y_averages = scores_mean #np.mean(final_scores_array, axis= 0)
        print(y_averages)
        # print(np.sum(final_scores_array, axis=0))
        y_mins = np.min(final_scores_array_scores, axis = 0)
        y_maxes = np.max(final_scores_array_scores, axis = 0) 
    
        y_errors_min = abs(y_averages - y_mins)
        y_errors_max = abs(y_averages - y_maxes)

        y_errors = np.concatenate((np.expand_dims(y_errors_min, axis=0), np.expand_dims(y_errors_max, axis=0)))
    

        ax.errorbar(x, y_averages, yerr = y_errors, capsize=5, fmt='ro--', ecolor='blue')#color=colours[colour_counter])
        ax.set_xticks(x)
        # colour_counter += 1
        # print(colour_counter)
        # colour_counter = 0
        # x = x[0]
        # for experiment_key in list(moving_averages.keys()):
            
        #     experiment_dict = moving_averages[experiment_key]
        #     update_keys = list(experiment_dict.keys())
        #     update_keys.sort(key=int)
        #     for update_key in update_keys:
        #         #print(list(experiment_dict.keys()))
        #         print(experiment_dict[update_key])
        #         ax.plot(x, experiment_dict[update_key], color = colours[colour_counter])
        #         colour_counter += 1
                
        
        #ax.legend(loc = 'below')
        # ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
        #             bbox_transform=fig.transFigure, ncol=4)
        ax.set_ylim(bottom=np.min(y_mins - 0.05), top=1.0)

        #ax.scatter(x, np.array(scores))

        
        plt.savefig(os.path.join(save_dir, "deepedit_iterations.png"))
        plt.show()
    else:
        x = np.array(range(len(final_scores_array_scores)))
        y = np.array(final_scores_array_scores)

        ax.set_ylim(bottom=max(0, np.min(y) - 0.2), top=min(np.max(y) + 0.2, 1)) #1.0)

        print(x.shape)
        print(y.shape)

        dataset = pd.DataFrame({
            "value":np.squeeze(y)
        })
        # ax.scatter(x, y)
        # plt.show()

        sb.swarmplot(data=dataset["value"], size=5)
        sb.boxplot(y="value", data=dataset, whis=2.0)
        
        plt.savefig(os.path.join(save_dir, "initialisation.png"))
        plt.show()
def _comparison_visualisation(scores_nested, task_configs, result_dirs, inference_image_subtasks, comparison_type, datetimes, checkpoints, plot_type, run_infer_string, guidance_set, rejection_value):
    import seaborn as sb
    import pandas as pd

    #This is for comparing across models and checkpoints. Not between frameworks. 


    inference_task = task_configs[1]
    #print(f'Inference task is {inference_task}')
    if inference_task == "deepedit":
        initialisation = task_configs[2]
        num_clicking_iters = task_configs[3] 
        task_title = f'{initialisation.capitalize()} initialisation, {num_clicking_iters} {inference_task.capitalize()} iterations'
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title(f'{task_title[:-1]}  scores')
        ax.set_xlabel('Clicking Iteration')
        ax.set_ylabel(' Score')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        task_title = inference_task.capitalize()
        ax.set_title(f'{inference_task.capitalize()}  Scores')
        ax.set_ylabel(' Score')
    
    #Tracking the absolute y_min for the error bars
    tracking_y_min = [1] * (int(num_clicking_iters) + 1)
    #Tracking the averages across all to determine the y_average min.
    y_averages_tracking = np.array([1] * (int(num_clicking_iters) + 1))


    colour_counter = 0
    dot_format = ["rx--", "bx--", "gx--", "cx--", "kx--", "mx--"] 
    c_map = ["blue", "green", "red", "purple", "orange", "black"]

    ########### Loading in the model summaries for the models ##############

    with open("/home/parhomesmaeili/Model_Config_Details/model_summary_details.txt") as f:
        dictionary = f.read()
        #print(dictionary)
        summary_details = json.loads(dictionary)

    #n_comparisons = len(result_dirs)
    #print(n_comparisons)

    ## Tracking all the failures across the different comparison models to examine them side-by-side ###
    all_failures = dict()

    for index, result_dir in enumerate(result_dirs):
        # If we are doing a deepedit output, then we just want a moving average across the iterations (possibly with some range bars?)

        # If we are doing an autoseg or deepgrow output, then we just want a distribution of the scores. 

        ################ Removing the failure cases from the mean computation, and printing the list of failure cases into a separate text file to use as exemplars. ###################
        #print(np.array(scores).shape)
        #print(result_dir)
        #print(inference_image_subtasks[index])

        scores = scores_nested[index]

        tmp_array = np.array(scores)

        scores_array = tmp_array[1:,:].astype(np.float64)
        scores_array = scores_array.T 

        
        scores_array = scores_array[:,:]
        image_names = tmp_array[0,:]

        #Setting failure case  score:
        failure_ = rejection_value
        #print(scores_array.shape)
        failure_images = dict()
        row_removal_ins = []
        #Going up until shape -2 because we do not want to examine the averages or the stdevs in the row removal phase
        for _row in range(scores_array.shape[0] - 2): #np.array(scores):
            sub_array = scores_array[_row, :]
            #print(sub_array)
            if np.any(sub_array < failure_):
                #print(sub_array)
                #print(image_names[index])

                failure_images[image_names[_row]] = sub_array.tolist()
                row_removal_ins.append(_row)

        #Adding the list of failures to the total comparison dict 
        #print(checkpoints)
        #print(datetimes)
        #print(index)
        if comparison_type == "checkpoint":
            all_failures[checkpoints[index]] = failure_images 
        elif comparison_type == "model":
            all_failures[datetimes[index]] = failure_images 
        # all_failures[inference_image_subtasks[index]] = failure_images

        #scores = [i if i >= failure_ in scores]
        #scores = []
        ################################################################################################################################################################################

        ######################### For the initialisation failures/general failures, save the names of the images #######################################################

        # with open(os.path.join(result_dir,'failure_cases_comparison_loop.txt'), 'w') as f:
        #     f.write(json.dumps(failure_images))
        

        ###### Remove the failure case rows ##########

        final_scores_array = np.delete(scores_array, row_removal_ins, axis=0)

        final_scores_array_scores = final_scores_array[:-2,:]
        scores_mean = final_scores_array[-2, :]
        scores_stdev = final_scores_array[-1,:]
        ####################################################################################################################################

        if inference_task == "deepedit":
            x = np.array(range(final_scores_array_scores.shape[1]))
            y_averages = scores_mean #np.mean(final_scores_array, axis= 0)
            #print(y_averages)
            y_mins = np.min(final_scores_array_scores, axis = 0)
            y_maxes = np.max(final_scores_array_scores, axis = 0) 
        
            y_errors_min = abs(y_averages - y_mins)
            y_errors_max = abs(y_averages - y_maxes)

            y_errors = np.concatenate((np.expand_dims(y_errors_min, axis=0), np.expand_dims(y_errors_max, axis=0)))

            if plot_type == "errorbar":
                if comparison_type == "checkpoint":
                    ax.errorbar(x, y_averages, yerr = y_errors, capsize=8, elinewidth=2, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=checkpoints[index])
                elif comparison_type == "model":
                    ax.errorbar(x, y_averages, yerr = y_errors, capsize=8, elinewidth=2, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=summary_details[datetimes[index]])
                # ax.errorbar(x, y_averages, yerr = y_errors, capsize=5, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=) #ecolor='blue')#color=colours[colour_counter])
            else:
                if comparison_type == "checkpoint":
                    ax.plot(x, y_averages, dot_format[colour_counter], label=checkpoints[index])
                elif comparison_type == "model":
                    ax.plot(x, y_averages, dot_format[colour_counter], label=summary_details[datetimes[index]])
                   
            
            # print(c_map(colour_counter))
            colour_counter += 1
            for ind, val in enumerate(y_mins.tolist()): # y_mins < tracking_y_min:
                if tracking_y_min[ind] > val:
                    tracking_y_min[ind] = val 

            y_averages_tracking = np.concatenate((y_averages_tracking, y_averages)) 

        #ax.scatter(x, np.array(scores))
    
    #Saving all the failure cases: 
    #For the comparisons we need to save them in a separate folder:
    
    #Base directory which we will place the folder in:

    if comparison_type == "checkpoint":
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(result_dir))), 'Plots', "Comparison", comparison_type)
        folder_save_name = f"model_{datetimes[0]}/checkpoints"
        for checkpoint in checkpoints:
            folder_save_name += f"_{checkpoint}"
    elif comparison_type == "model":
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(result_dir)))), 'Plots', "Comparison", comparison_type)
        folder_save_name = f"checkpoint_{checkpoints[0]}/models"
        for model in datetimes:
            folder_save_name += f"_{model}"

    save_dir = os.path.join(base_dir, f'{folder_save_name}/{run_infer_string}/{guidance_set}')
    os.makedirs(save_dir, exist_ok=True)
    #Saving the failure cases across the comparison sets

    with open(os.path.join(save_dir,'failure_cases_comparison_loop.txt'), 'w') as f:
            f.write(json.dumps(all_failures))

    if inference_task == "deepedit":
        ax.set_xticks(x)
        # colour_counter += 1
        # print(colour_counter)
        # colour_counter = 0
        # x = x[0]
        # for experiment_key in list(moving_averages.keys()):
            
        #     experiment_dict = moving_averages[experiment_key]
        #     update_keys = list(experiment_dict.keys())
        #     update_keys.sort(key=int)
        #     for update_key in update_keys:
        #         #print(list(experiment_dict.keys()))
        #         print(experiment_dict[update_key])
        #         ax.plot(x, experiment_dict[update_key], color = colours[colour_counter])
        #         colour_counter += 1
                
        
        #ax.legend(loc = 'below')
        ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
                    bbox_transform=fig.transFigure, ncol=4)
        
        if plot_type == "errorbar":
            ax.set_ylim(bottom=np.min(np.array(tracking_y_min) - 0.05), top=1.0)
        elif plot_type == "scatter":
            ax.set_ylim(bottom=max(0,np.min(y_averages_tracking) - 0.05), top=min(1, np.max(y_averages_tracking) + 0.05))

    
    plt.savefig(os.path.join(save_dir, "comparison_plot.png"), bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/imagesTs")
    parser.add_argument("--datetime", nargs="+", default=["31052024_195641"])
    parser.add_argument("--checkpoint", nargs="+", default=["best_val_score_epoch"])
    parser.add_argument("--infer_run", nargs="+", default=['0', '0', '0'])
    parser.add_argument("-ta", "--task", nargs="+", default=["deepeditplusplus","deepedit", "autoseg", "3"], help="The framework selection + subtask/mode which we want to execute")
    parser.add_argument("--app_dir", default = "MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus")
    parser.add_argument("--job", default= "collecting", help="argument that determines which job is required from the script (i.e. plotting or computing  scores)")
    parser.add_argument("--plot_type", default="errorbar")
    # parser.add_argument("--")
    parser.add_argument("--human_measure", nargs='+', default=["locality", "original_guidance"], help='the additional parameters contain the information about which set of guidance points for which we are plotting  scores')
    parser.add_argument("--metric_type", default="distance", help='the type of metric being considered, a cross section, a binary based on ellipsoid/cuboid or a distance weighted mask based metric')
    parser.add_argument("--metric_base", help='The base form of the metric under consideration, which will have masking applied to it. E.g.  score, error rate etc.')
    parser.add_argument("--roi_scale_size", default='0.025')
    parser.add_argument("--rejection_val", default='0.5', help='Parameter which controls what the failure value is for the  scores to not include in averages')
    
    #parser.add_argument("--models")
    args = parser.parse_args()

    app_dir = os.path.join(up(up(up(os.path.abspath(__file__)))), args.app_dir)
    framework = args.task[0]
    inference_task = args.task[1]
    
    dataset_name = args.studies[:-9]
    dataset_subset = args.studies[-8:]

    # #This is for the single checkpoint, single model, single run inference output processing.
    # if len(args.datetime) == 1 and len(args.checkpoint) == 1 and len(args.infer_run) == 1:
    #     datetime = args.datetime[0]
    #     checkpoint = args.checkpoint[0]

    #     if inference_task == "deepedit":
    #         initialisation = args.task[2]
    #         num_clicking_iters = args.task[3] 

    #         inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_{args.infer_run[0]}'
    #         inference_image_subdirectory = 'datasets/' + inference_image_subtask
    #     else:
    #         inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_{args.infer_run[0]}'
    #         inference_image_subdirectory = 'datasets/' + inference_image_subtask

    #     results_save_dir = app_dir + f'/results_{args.human_measure[0]}/' + inference_image_subtask 
    #     #print(results_save_dir)
    #     os.makedirs(results_save_dir, exist_ok=True)

    job = args.job
    
    #This is for computing the  scores for a single run of a single model/single checkpoint.
    if job == "compute":

        #This is for the single checkpoint, single model, single run inference output processing.
        if len(args.datetime) == 1 and len(args.checkpoint) == 1 and len(args.infer_run) == 1:
            datetime = args.datetime[0]
            checkpoint = args.checkpoint[0]
            
            run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3] 

                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/{run_infer_string}'
                inference_image_subdirectory = 'datasets/' + inference_image_subtask
            else:
                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/{run_infer_string}'
                inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + f'/results_{args.human_measure[0]}_{args.metric_type}_{args.metric_base}/' + inference_image_subtask 
            #print(results_save_dir)
            # os.makedirs(results_save_dir, exist_ok=True)
        
        score_computation(os.path.join(app_dir, inference_image_subdirectory), args.task, results_save_dir, job, args.studies, args.human_measure[0], args.metric_type, args.metric_base, float(args.roi_scale_size))

    #This is for collecting the  scores across the different runs of a single model/single checkpoint.
    elif job == "collecting":

        datetime = args.datetime[0]
        checkpoint = args.checkpoint[0]
        run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

        if inference_task == "deepedit":
            initialisation = args.task[2]
            num_clicking_iters = args.task[3]
            human_measure = args.human_measure

            inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_{infer_run}/{human_measure[1]}' for infer_run in args.infer_run]
            # inference_image_subdirectory = 'datasets/' + inference_image_subtask
            results_save_dir = app_dir + f'/results_{human_measure[0]}_{args.metric_type}_{args.metric_base}/' + dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_collection/{run_infer_string}/{human_measure[1]}'
        else:

            human_measure = args.human_measure 

            inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_{infer_run}/{human_measure[1]}' for infer_run in args.infer_run] 
            # inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + f'/results_{human_measure[0]}_{args.metric_type}_{args.metric_base}/' + dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_collection/{run_infer_string}/{human_measure[1]}'
        

        score_files_base_dir = app_dir + f'/results_{human_measure[0]}_{args.metric_type}_{args.metric_base}/'
        # #print(results_save_dir)
        # if os.path.exists(results_save_dir):
        #     shutil.rmtree(results_save_dir)

        # os.makedirs(results_save_dir, exist_ok=True)

        score_collection(results_save_dir, inference_image_subtasks, score_files_base_dir, float(args.rejection_val))#, args.infer_run) 


    #This is for plotting a single RUN single MODEL/single checkpoint or concatenated runs single model/single checkpoint
    elif job == "plot_single_model":
        #This is for the single checkpoint, single model, single run inference output processing.
        if len(args.datetime) == 1 and len(args.checkpoint) == 1:
            datetime = args.datetime[0]
            checkpoint = args.checkpoint[0]

            #Here we will convert the list of runs into a single string which outlines the folder which we our collated  score results are stored in.
            run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3] 

                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
                # inference_image_subdirectory = 'datasets/' + inference_image_subtask
            else:
                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
                # inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + f'/results_{args.human_measure[0]}_{args.metric_type}_{args.metric_base}/' + inference_image_subtask 
            #print(results_save_dir)
            # os.makedirs(results_save_dir, exist_ok=True)

        
        results_save_dir = os.path.join(results_save_dir, args.human_measure[1])
        scores = score_extraction(results_save_dir) #score_computation(os.path.join(app_dir, inference_image_subdirectory), args.task, results_save_dir, job, args.studies)
        
        #Here we split the row containing the averages from the rows containing the raw results.

        _visualisation(scores, args.task, results_save_dir, run_infer_string, args.human_measure[1], float(args.rejection_val))
    
    #This is for plotting a single RUN multiple model/checkpoint OR for the concatenated results across runs for multiple models/checkpoints
    elif job == "plot_multiple":
        results_save_base_dir = app_dir + f'/results_{args.human_measure[0]}_{args.metric_type}_{args.metric_base}/'
        #Here we will convert the list of runs into a single string which outlines the folder which we our collated  score results are stored in.
        run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])
        
        if len(args.datetime) > 1:
            # dataset_name = args.studies[:-9]
            # dataset_subset = args.studies[-8:]

            #In this case we are plotting different models entirely against one another
            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3]
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{args.checkpoint[0]}/run_collection/{run_infer_string}' for datetime in args.datetime]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]

            else:
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{args.checkpoint[0]}/run_collection/{run_infer_string}' for datetime in args.datetime]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks] 
            #Comparison subtype is the model setting
            comparison_subtype = "model"

        elif len(args.checkpoint) > 1:
            #In this case then we are plotting different checkpoints against one another
            # dataset_name = args.studies[:-9]
            # dataset_subset = args.studies[-8:]

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3]
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{args.datetime[0]}/{checkpoint}/run_collection/{run_infer_string}' for checkpoint in args.checkpoint]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]
            else:
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{args.datetime[0]}/{checkpoint}/run_collection/{run_infer_string}' for checkpoint in args.checkpoint]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]
            
            ##### Comparison subtype is : checkpoint
            comparison_subtype = "checkpoint"
            
        
        #Create the list of save directories to look into!
        results_save_dirs = [results_save_base_dir + f"{inference_image_subtask}/{args.human_measure[1]}" for inference_image_subtask in inference_image_subtasks]

        #print(results_save_dirs)
        #print(inference_image_subdirectories)

        scores = []
        #Saving a nested list of  scores
        for index, results_save_dir in enumerate(results_save_dirs):
            # os.makedirs(results_save_dir, exist_ok=True)
            #print(inference_image_subdirectories[index])
            scores.append(score_extraction(results_save_dir)) #score_computation(os.path.join(app_dir, inference_image_subdirectories[index]), args.task, results_save_dirs[index], job, args.studies))
        
        _comparison_visualisation(scores, args.task, results_save_dirs, inference_image_subtasks, comparison_subtype, args.datetime, args.checkpoint, args.plot_type, run_infer_string, args.human_measure[1], float(args.rejection_val))