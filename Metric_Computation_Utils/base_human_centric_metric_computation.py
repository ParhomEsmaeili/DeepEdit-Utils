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
from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose
            )
# from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd  
# from monai.metrics import Helper
# from monai.metrics import Metric
# from monai.utils import MetricReduction
# import torch


utils_codebase__dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_codebase__dir)
# mask_generator_dir = os.path.join(os.path.abspath(__file__), 'Mask_Generation_Utils')

from Metric_Computation_Utils.scoring_base_utils import ScoreUtils 
from Mask_Generation_Utils.metric_mask_generator_utils import MaskGenerator 

class score_tool():
    def __init__(self, 
                 label_names, 
                 human_measure, 
                 click_weightmap_types, 
                 gt_weightmap_types, 
                 metric_base, 
                 include_background_mask, 
                 include_background_metric, 
                 ignore_empty, 
                 include_per_class_scores):
    
        self.human_measure = human_measure 
        self.click_weightmap_types = click_weightmap_types
        self.gt_weightmap_types = gt_weightmap_types
        self.metric_base = metric_base 
        # self.mask_parametrisations = mask_parametrisations
        # self.image_size = image_size 
        self.dict_class_codes = label_names 
        self.include_background_mask = include_background_mask 
        self.include_background_metric = include_background_metric
        self.ignore_empty = ignore_empty
        self.include_per_class_scores = include_per_class_scores

        '''
        Our assumption will be that the ground truths and the class label codes are already pre-matching.
        '''
        if self.human_measure.title() == "Local Responsiveness" or self.human_measure.title() == "None":
            self.transforms_list = [
            LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred", "gt")),
            Orientationd(keys=("pred", "gt"), axcodes="RAS")
            ]

        if self.human_measure == "Temporal Non Worsening":
            self.transforms_list = [
            LoadImaged(keys=("pred_1", "pred_2", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred_1","pred_2", "gt")),
            Orientationd(keys=("pred_1","pred_2", "gt"), axcodes="RAS")
            ]  

        if self.human_measure == "Temporal Consistency": #In this case there is no comparison to a ground truth. The "GT" in the metric will be the prior predictions.
            self.transforms_list = [
            LoadImaged(keys=("pred_1", "pred_2"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred_1", "pred_2")),
            Orientationd(keys=("pred_1", "pred_2"), axcodes="RAS")
            ]

        self.transforms_composition = Compose(self.transforms_list, map_items = False)
    
        self.supported_click_weightmaps = ['Ellipsoid',
                                            'Cuboid', 
                                            'Scaled Euclidean Distance',
                                            'Exponentialised Scaled Euclidean Distance',
                                            'Binarised Exponentialised Scaled Euclidean Distance'
                                            '2D Intersections', 
                                            'None']
        
        self.supported_gt_weightmaps = ['Connected Component',
                                        'None']
            
        self.supported_human_measures_dice = ['Local Responsiveness',
                                        'Temporal Consistency', #Difference between temporal consistency and temporal non worsening is that the former
                                        #is not filtered by the changed voxels pre and post-click.

                                        'None'] #In this context, None = Default global dice score.
        
        self.supported_human_measures_error_rate = ['Temporal Non Worsening',
                                                    ]

        if any([weightmap not in self.supported_click_weightmaps for weightmap in self.click_weightmap_types]):
            raise ValueError("The selected click weightmap type is not supported by the mask generator utilities")
        
        if any([weightmap not in self.supported_gt_weightmaps for weightmap in self.gt_weightmap_types]):
            raise ValueError("The selected gt weightmap type is not supported by the mask generator utilities") 
        
        if self.metric_base == 'Dice':

            if self.human_measure not in self.supported_human_measures_dice:
                raise ValueError("The selected human measure is not intended or supported by the mask generator utility")
        
        elif self.metric_base == 'Error Rate':

            if self.human_measure not in self.supported_human_measures_error_rate:
                raise ValueError("The selected human measure is not intended or supported by the mask generator utility")

        # #Extract the corresponding weightmaps for the corresponding subtypes.

        # if all(self.weightmap_types) in self.supported_click_weightmaps:
        
        #     self.click_weightmaps_types = self.weightmap_types
        #     self.gt_weightmap_types = ['None']

        # elif all(self.weightmap_types) in self.supported_gt_weightmaps:
        
        #     self.click_weightmap_types = ['None']
        #     self.gt_weightmap_types = self.weightmap_types
        
        # elif self.weightmap_types == ['None']:

        #     self.click_weightmap_types = ['None']
        #     self.gt_weightmap_types = ['None']

        # else:

        #     #If there is any weightmap that is in the click weightmap types, then extract those
        #     self.click_weightmap_types = [subtype.title() for subtype in self.weightmap_types if subtype.title() in self.supported_click_weightmaps]
        #     self.gt_weightmap_types = [subtype.title() for subtype in self.weightmap_types if subtype.title() in self.supported_gt_weightmaps]


        #Here we initialise the mask generator class:

        ignore_empty_mask = True #We want our mask generation be capable of generating a nan tensor when the click set (which may be required for click-centric metrics) is not available for some reason!

        self.mask_generator = MaskGenerator(self.click_weightmap_types, self.gt_weightmap_types, human_measure, self.dict_class_codes, ignore_empty_mask)

        #Defining a class with the scoring utils, which can incorporate masks when computing scores. 
        self.scoring_util = ScoreUtils(self.metric_base, self.include_background_metric, self.ignore_empty, self.include_per_class_scores, self.dict_class_codes)

    def __call__(self, pred_folder_paths, gt_folder_path, image_name, guidance_points_set, guidance_points_parametrisations):
        
        '''
        #Here the guidance points set is a dictionary for the current segmentation (and image) under consideration. This dictionary contains the list of points for each
        #class for the current iteration of segmentation. 

        The guidance points parametrisations are the parametrisations that correspond to the generation of the image masks. This is divided by mask types, class names, and points by dict, dict, list

        #We assume here that the image name has the file type extension.


        #We allow for multiple prediction folder paths, since temporal consistency metrics require it.

        '''
        
        assert type(pred_folder_paths) == list 
        assert type(gt_folder_path) == str or gt_folder_path == None 
        assert type(image_name) == str 
        assert type(guidance_points_set) == dict
        assert type(guidance_points_parametrisations) == dict  
        
        pred_image_paths = [os.path.join(pred_folder_path, image_name) for pred_folder_path in pred_folder_paths]
        gt_image_path = os.path.join(gt_folder_path, image_name) if gt_folder_path != None else None 

        if self.human_measure == "Local Responsiveness" or self.human_measure == "None":
            assert gt_folder_path != None 

            if self.include_background_metric:
                assert self.include_background_mask == True 
            else:
                pass 

            input_dict = {"pred":pred_image_paths[0], "gt":gt_image_path}
            output_dict = self.transforms_composition(input_dict)

            #Output dict will contain the extracted pred(s) and gt in RAS orientation. 
            final_pred = torch.tensor(output_dict['pred'][0])
            gt = torch.tensor(output_dict['gt'][0])

            human_measure_information = None 
            
            #We extract the first channel since it is a batchwise function.

        elif self.human_measure == "Temporal Non Worsening":
            assert gt_folder_path != None
            #We place an initial assertion that the following configurations are permitted for the error rate generation:

            if self.include_background_metric == True:
                
                assert self.include_background_mask == True  #We must do this because otherwise there would not be a background click/gt weightmap to correspond to 
            else:
                pass #Else, nothing needs to be checked, either configuration will be acceptable for the background mask param. If it was true, then it would just be ignored anyways.
            

            input_dict = {"pred_1":pred_image_paths[0], "pred_2":pred_image_paths[1], "gt":gt_image_path}
            output_dict = self.transforms_composition(input_dict)

            #Output dict will contain the extracted pred(s) and gt in RAS orientation. 
            pred_1 = torch.tensor(output_dict['pred_1'][0])
            pred_2 = torch.tensor(output_dict['pred_2'][0])
            gt = torch.tensor(output_dict['gt'][0])

            final_pred = pred_2
            
            try:
                background_class_code = self.dict_class_codes['background']
            except:
                background_class_code = self.dict_class_codes['Background']
            

            #Human measure information for this  will contain a set of bool masks which pass forward information with respect to whether the background is included for 
            # metric computation.
            human_measure_information = dict()

            #We use the metric background bool because the human measure information is a weighting which is intended to be pre-applied to the metric computation
            #in order to drop out the non-changed voxels. Correspondingly, if background is not being desired in the metric, the weightmap should reflect this. 
            if self.include_background_metric:

                cross_class_changed_voxels = torch.where(pred_1 != pred_2, 1, 0)
                human_measure_information['cross_class_changed_voxels'] = cross_class_changed_voxels
            
            else:
                
                #Extracting the non background voxels from the ground truth to mask the changed voxels which belong to non-background since these shouldn't be used in the metric
                #I.e. we are only interested in the changed voxels which are non background gt. Very unlikely..
                cross_class_changed_voxels = torch.where(pred_1 != pred_2, 1, 0)

                human_measure_information['cross_class_changed_voxels'] = cross_class_changed_voxels * torch.where(gt != background_class_code, 1, 0)


            if self.include_per_class_scores:
                
                per_class_changed_voxels = dict()

                for class_label, class_code in self.dict_class_codes.items():
                    
                    if class_label.title() == 'Background':
                        if not self.include_background_metric:
                            continue
                        else:
                            pred_1_per_class = torch.where(pred_1 == int(class_code), 1, 0)

                            pred_2_per_class = torch.where(pred_2 == int(class_code), 1, 0)

                            gt_per_class = torch.where(gt == int(class_code), 1, 0)
                            # if self.include_background_metric:

                            #For each class, we just want to extract the corresponding changed voxels for that given class only.

                            per_class_changed_voxels[class_label] = torch.where(pred_1_per_class != pred_2_per_class, 1, 0) * gt_per_class 

                    else:

                        pred_1_per_class = torch.where(pred_1 == int(class_code), 1, 0)

                        pred_2_per_class = torch.where(pred_2 == int(class_code), 1, 0)

                        gt_per_class = torch.where(gt == int(class_code), 1, 0)
                        # if self.include_background_metric:

                        #For each class, we just want to extract the corresponding changed voxels for that given class only.

                        per_class_changed_voxels[class_label] = torch.where(pred_1_per_class != pred_2_per_class, 1, 0) * gt_per_class 

                
                human_measure_information['per_class_changed_voxels'] = per_class_changed_voxels

            
        elif self.human_measure == "Temporal Consistency":
            
            assert gt_folder_path == None 

            if self.include_background_metric:
                assert self.include_background_mask == True 
            else:
                pass 

            input_dict = {"pred_1":pred_image_paths[0], "pred_2":pred_image_paths[1]}
            output_dict = self.transforms_composition(input_dict)

            #Output dict will contain the extracted pred(s) in RAS orientation. 
            final_pred = torch.tensor(output_dict['pred_2'][0])
            gt = torch.tensor(output_dict['pred_1'][0])

            human_measure_information = None 
            
            #We extract the first channel since it is a batchwise function.


        #We extract the image_size dimensions here in RAS orientation.
        
        image_dims = gt.size()

        cross_class_map, per_class_maps = self.mask_generator(guidance_points_set, guidance_points_parametrisations, self.include_background_mask, human_measure_information, image_dims, gt)
        
        output_score = self.scoring_util(image_masks=(cross_class_map, per_class_maps), pred=final_pred,gt=gt)
        
        #Output score should be a dict, with one key:val pair corresponding to the cross-class score, and another containing a dict with the class-separated scores.

        assert type(output_score) == dict
        assert type(output_score['overall score']) == torch.Tensor 
        assert type(output_score['per class scores']) == dict 
        

        return output_score['overall score'], output_score['per class scores']

    
    




