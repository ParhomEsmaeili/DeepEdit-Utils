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

file_dir = up(os.getcwd())
sys.path.append(file_dir)

from scoring_base_utils import ScoreUtils 

class score_tool():
    def __init__(self, label_names, human_measure, metric_type, metric_base, metric_click_size):
    
        from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose
            )
        # from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd  
        # from monai.metrics import Helper
        # from monai.metrics import Metric
        from monai.utils import MetricReduction
        # import torch

    
        self.human_measure = human_measure 
        self.metric_type = metric_type 
        self.metric_base = metric_base 
        self.metric_click_size = metric_click_size
        # self.image_size = image_size 
        self.label_names = label_names 

        #Here we denote some pre-transforms for the  score computations. The same transforms list is not used because for LOCALITY the GT needs to be mapped 
        #to the same labels as the segmentation. For temporal consistency it is also required because the guidance points are stored using the labels being mapped to
        #as the keys in the dictionary storing them, but this is provided by self.label_names.

        if self.human_measure == "locality":
            self.transforms_list = [
            LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred", "gt")),
            Orientationd(keys=("pred", "gt"), axcodes="RAS")
            ]

        if self.human_measure == "temporal_consistency":
            self.transforms_list = [
            LoadImaged(keys=("pred_1", "pred_2", "gt"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("pred_1","pred_2", "gt")),
            Orientationd(keys=("pred_1","pred_2", "gt"), axcodes="RAS")
            ]  

        self.transforms_composition = Compose(self.transforms_list, map_items = False)
        #Defining a class with the scoring utils, which can incorporate masks when computing scores. 
        self.scoring_util = ScoreUtils(metric_base)


    def __call__(self, pred_folder_paths, gt_folder_path, image_name, guidance_points_set):

        #Here the guidance points set is a dictionary for the current segmentation under consideration. This dictionary contains the list of points for each
        #class for the current iteration of segmentation. 

        #We allow for multiple prediction folder paths, since temporal consistency metrics require it.

        
        pred_image_paths = [os.path.join(pred_folder_path, image_name) for pred_folder_path in pred_folder_paths]
        gt_image_path = os.path.join(gt_folder_path, image_name)

        if self.human_measure == "locality":
            
            input_dict = {"pred":pred_image_paths[0], "gt":gt_image_path}

        elif self.human_measure == "temporal_consistency":

            input_dict = {"pred_1":pred_image_paths[0], "pred_2":pred_image_paths[1], "gt":gt_image_path}

        
        output_dict = self.transforms_composition(input_dict)

        #We extract the image_size dimensions here because we want to implement the boxing of the guidance points in the RAS orientation. Therefore we need
        #to obtain the image size in that orientation so that we can have the appropriate box dimensions also.
        
        self.image_size = output_dict["pred"].shape[1:]

        image_mask = torch.Tensor(self.mask_generator(guidance_points_set))

        #We need to split the pred and GT into a class-by-class tensor EXCEPT for the background class. Any change for the background class is reflected in the other classes!

        #We also do not want to "dilute" the metric computation by using the background class which can heavily bias it to higher scores.

        
        #Here we preapply our mask to each one-hot class segmentation. Compute the overlaps, and then compute the metric across all the classes.  

        
        output_score = self.scoring_util(ignore_empty=True, image_mask=image_mask, pred=output_dict["pred"], gt=output_dict["gt"])
        
        return output_score

    
    def mask_apply(self, guidance_points_set, roi_size, ellipsoid_shape_bool):
        #We will not be extracting separate masks for each class because the intention should be that whatever desired characteristic is being exhibited, should
        #be doing so across all classes. AND even if a click is being placed for a background point, then it should be reflected in an improvement to the other classes...?

        #Initialise a tensor extracting the regions around the guidance points.

        mask_tensor = np.zeros(self.image_size)

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
                        mask_tensor[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1]] = 1
                    elif len(self.image_size) == 3:
                        #If image is 3D:
                        mask_tensor[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1], min_maxes[2][0]:min_maxes[2][1]] = 1
                
        return mask_tensor
    




