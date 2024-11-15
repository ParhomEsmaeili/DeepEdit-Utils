'''
This script is for the simple act of performing per-sample averaging of a set of merged results
'''

import os
import json 
from pathlib import Path 
import torch 
import numpy as np
import csv 
import sys
from os.path import dirname as up
utils_dir = os.path.join(up(up(os.path.abspath(__file__))))
sys.path.append(utils_dir)
from Metric_Computation_Utils.score_generation_path_utils import path_generation 
import shutil 
import math 


class sample_averaged_score_generator():
    def __init__(self, args):
        
        assert type(args) == dict, 'Score generation failed because the config was not a dict'


        #TODO: ADD assertions for each of the fields in the inputs correspondingly. 
        self.dataset_subset = args['dataset_subset']
        # self.sequentiality_mode = args['sequentiality_mode']
        # self.ignore_empty = args['ignore_empty']
        self.per_class_scores = args['per_class_scores']
        # self.include_background_mask = args['include_background_mask']
        self.include_background_metric = args['include_background_metric']
        self.app_dir_path = os.path.join(os.path.expanduser("~"), args['app_dir'])
        self.infer_run_mode = args['inference_run_mode']
        self.human_measure = args['human_measure']
        self.base_metric = args['base_metric']
        self.gt_weightmap_types = args['gt_weightmap_types']
        self.click_weightmaps_dict = args['click_weightmap_dict']
        self.infer_run_parametrisation = args['inference_run_parametrisation'] #This parametrisation pertains to both the click size but also to whether it is working in CIM/1-iter SIM type modes
        self.infer_run_nums = args['inference_run_nums']
        self.infer_simulation_type = args['simulation_type']
        self.checkpoint = args['checkpoint']
        self.datetime = args['datetime']
        self.studies = args['studies'] 
        self.include_nan = args['include_nan']
        self.num_samples = args['num_samples']
        self.total_samples = args['total_samples']


        assert type(self.dataset_subset) == str 
        # assert type(self.sequentiality_mode) == str 
        # assert type(self.ignore_empty) == bool 
        assert type(self.per_class_scores) == bool 
        # assert type(self.include_background_mask) == bool 
        assert type(self.include_background_metric) == bool 
        assert type(self.app_dir_path) == str 
        assert type(self.infer_run_mode) == list 
        assert len(self.infer_run_mode) == 3, "This script is only intended for multi-iteration score computation"
        assert type(self.human_measure) == str 
        assert type(self.base_metric) == str
        assert type(self.gt_weightmap_types) == list 
        assert type(self.click_weightmaps_dict) == dict
        assert type(self.infer_run_parametrisation) == dict
        assert type(self.infer_run_nums) == list 
        assert type(self.infer_simulation_type) == str
        assert type(self.checkpoint) == str 
        assert type(self.datetime) == str
        assert type(self.studies) == str 
        assert type(self.include_nan) == bool
        # assert type(self.summary_dict) == dict 

    def supported_configs(self): 

        supported_initialisations = ['Autoseg','Interactive']
        
        supported_click_weightmaps =['Ellipsoid',
                                    'Cuboid', 
                                    'Scaled Euclidean Distance',
                                    'Exponentialised Scaled Euclidean Distance',
                                    'Binarised Exponentialised Scaled Euclidean Distance',
                                    '2D Intersections', 
                                    'None']
        
        supported_gt_weightmaps = ['Connected Component',
                                    'None']
            
        supported_human_measures = ['Local Responsiveness',
                                    'Temporal Non Worsening',
                                    'Temporal Consistency',
                                    'None'] 
        supported_base_metrics = ['Dice',
                                'Error Rate']
        
        supported_score_summaries = ['Mean',
                                     'Median',
                                     'Standard Deviation',
                                     'Interquartile Range',
                                     'Lower Quartile',
                                     'Upper Quartile',
                                     'Minimum',
                                     'Maximum']
        
        supported_simulation_types = ['probabilistic',
                                      'deterministic']
        '''
        Corresponding parametrisations:

        click weightmaps:

        none: none 

        gt_weightmaps:
        
        none: none 

        '''
        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_simulation_types
    
    def score_extraction(self, results_save_dir, metric):
        
        assert type(results_save_dir) == str 
        assert os.path.exists(results_save_dir) 

        #obtaining the paths for all of the score files we want to merge together:
        score_path = os.path.join(results_save_dir, f'{metric}_score_results.csv')

        assert os.path.exists(score_path) 

        num_experiment_repeats = len(self.infer_run_nums) 

        valid_sample_indices = [j for sublist in [list(range(self.total_samples * i,  self.total_samples * i + self.num_samples)) for i in range(num_experiment_repeats)] for j in sublist]


        #extracting the scores
        
        with open(score_path, newline='') as f:
            score_reader = csv.reader(f, delimiter=' ', quotechar='|')
            first_row = f.readline()
            first_row = first_row.strip()
            
            scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] 

            for row in score_reader:


                row_str_list = row[0].split(',')
                
                for index, string in enumerate(row_str_list):
                    if index > 0:
                        scores[index].append(float(string))
                    elif index == 0:
                        scores[index].append(string)
            
            #Read all of the results, then just keep the valid indices. 

        output_scores = [scores[0]]
        
         
        for sublist in scores[1:]:
            
            valid_sublist = [val for i, val in enumerate(sublist) if i in valid_sample_indices]  #and not math.isnan(val)]
            output_scores.append(valid_sublist)

        #Formatting of output scores is a nested list, the first sublist is a list of names of the image samples. Then each successive sublist is the corresponding list of scores
        #for each iteration in the segmentation output. 

        return output_scores 

    def per_sample_averaging(self, scores):
        
        num_experiments = len(self.infer_run_nums)

        #we assume the image names are still there in the first index! 
        output = [scores[0]] 

        #We filter the nan scores when computing averages, if for each sample there is not a non-nan score then just continue..

        for sublist in scores[1:]:
            
            current_iter_averaged = [] 

            for index in range(self.num_samples):
                experiment_values = [sublist[j * self.num_samples + index] for j in range(num_experiments)]
                
                if not self.include_nan:
                    non_nan_vals = [val for val in experiment_values if not math.isnan(float(val))]
                    if len(non_nan_vals)  == 0:
                        #in this case just put a placeholder nan 
                        current_iter_averaged.append(float('nan'))
                    else:
                        #in this case, we used the filtered out nan values and average.
                        per_sample_mean = np.mean(non_nan_vals)
                        current_iter_averaged.append(per_sample_mean)

            #We then append the per sample averaged scores for that iteration.    
            output.append(current_iter_averaged)

        return output 

    def score_saving(self, results_dir, filename_base, scores):
        
        just_scores = scores[1:] #Nested list of per iteration scores.
        
        #Saving the new scores.
        
        for i in range(self.num_samples):
            #Extracting the sample name
            row = [scores[0][i]]
            
            score_list = [sublist[i] for sublist in just_scores]
            
            for val in score_list:
                row.append(val)
            
            with open(os.path.join(results_dir, filename_base),'a') as f:
                
                writer = csv.writer(f)
                writer.writerow(row) 
    
    def __call__(self):
    
        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode
        
        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset + f'_{self.infer_simulation_type}'
        
        inference_config_dict['datetime'] = self.datetime
        inference_config_dict['checkpoint'] = self.checkpoint

        inference_config_dict['inference_click_parametrisation'] = self.infer_run_parametrisation 

        assert type(self.infer_run_nums) == list 

        inference_config_dict['run_infer_string'] = 'run' + "".join([f"_{run}" for run in self.infer_run_nums])
        
        metric_config_dict = dict() 
        metric_config_dict['click_weightmap_types'] = list(self.click_weightmaps_dict.keys())
        
        #The click parametrisations dict is re-assigned, which we will later save alongside the computed metrics in their corresponding folders for reference later.    
        metric_config_dict['click_weightmap_parametrisations'] = self.click_weightmaps_dict 
        

        #Verifying that the selected configurations are supported by the downstream utilities.

        supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_simulation_types = self.supported_configs()


        if any([weightmap not in supported_click_weightmaps for weightmap in self.click_weightmaps_dict.keys()]):
            raise ValueError("The selected click weightmap types are not currently supported")
        
        if any([weightmap not in supported_gt_weightmaps for weightmap in self.gt_weightmap_types]):
            raise ValueError("The selected gt weightmap types are not currently supported")

        if self.human_measure not in supported_human_measures:
            raise ValueError("The selected human measure is not currently supported")
        
        if len(self.infer_run_mode) == 1:
            if self.infer_run_mode[0] not in supported_initialisations:
                raise ValueError("The selected initialisation strategy was not supported")
        else:
            if self.infer_run_mode[1] not in supported_initialisations:
                raise ValueError("The selected initialisation strategy was not supported")

        if self.base_metric not in supported_base_metrics:
            raise ValueError("The selected base metric was not supported")
    
        
        if self.infer_simulation_type not in supported_simulation_types:

            raise ValueError("The selected simulation type (e.g. probabilistic) was not supported")
        
        metric_config_dict['gt_weightmap_types'] = self.gt_weightmap_types
        metric_config_dict['human_measure'] = self.human_measure 
        metric_config_dict['base_metric'] = self.base_metric 

        ################################################################################################################################################################################

        #Generation of the paths required for extracting the (segmentations, guidance point sets, guidance point parametrisations etc.) and the path for saving the results.

        path_generation_class = path_generation(inference_config_dict, metric_config_dict)

        _, results_save_dir = path_generation_class()
        
        #We extract the dictionary of class-label - class-code correspondences. This should be located in the upper folder for the dataset at hand.

        label_config_path = os.path.join(self.app_dir_path, 'datasets', self.studies, 'label_configs.txt')
        
        assert os.path.exists(label_config_path)
        ################### Importing the class label configs dictionary #####################

        with open(label_config_path) as f:
            class_config_dict = json.load(f)

        config_labels = class_config_dict["labels"]

        #Summarisation config should also be saved as a json for checking:
        

        #Creating the folder which the per-sample averaged scores are saved to.
        results_per_sample_averaged_dir = os.path.join(results_save_dir, 'per_sample_averaged_results')

        os.makedirs(results_per_sample_averaged_dir, exist_ok=True)

        extracted_scores = self.score_extraction(os.path.join(results_save_dir, 'raw_results'), self.base_metric)

        filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)

        self.score_saving(results_per_sample_averaged_dir, f'{self.base_metric}_per_sample_averaged_results.csv', filtered_and_sample_averaged)

        if self.per_class_scores:

            for class_label in config_labels.keys():
                
                if not self.include_background_metric:
                    if class_label.title() == "Background":
                        continue 
                
                extracted_scores = self.score_extraction(os.path.join(results_save_dir, 'raw_results'), f'class_{class_label}_{self.base_metric}')
                
                filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)
                
                self.score_saving(results_per_sample_averaged_dir, f'class_{class_label}_{self.base_metric}_per_sample_averaged_results.csv', filtered_and_sample_averaged)