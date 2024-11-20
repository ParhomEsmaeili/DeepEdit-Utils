import numpy as np
import matplotlib.pyplot as plt
import os
import json 
# from pathlib import Path 
# import torch 
import numpy as np
import csv 
from os.path import dirname as up
import sys 

utils_dir = up(up(up(os.path.abspath(__file__))))
sys.path.append(utils_dir)

from Metric_Computation_Utils.score_generation_path_utils import path_generation 
# import shutil 
# import math 


'''Script which is used to generate a plotted trend against iterations for a given run/metric and summarisation set of summarisation statistics
ONLY for the relative dice score metrics.'''


class plot_trend:
    
    def __init__(self, args):
        
        if not type(args) == dict:
            raise TypeError('Score generation failed because the score merging config was not a dict')

        self.dataset_subset = args['dataset_subset']
        # self.sequentiality_mode = args['sequentiality_mode']
        # self.ignore_empty = args['ignore_empty']
        # self.per_class_scores = args['per_class_scores']
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
        # self.include_nan = args['include_nan']
        self.summary_dict = args['summary_dict']


        if not type(self.dataset_subset) == str:
            raise TypeError 
        # assert type(self.sequentiality_mode) == str 
        # assert type(self.ignore_empty) == bool 
        # assert type(self.per_class_scores) == bool 
        if not type(self.include_background_metric) == bool:
            raise TypeError
        if not type(self.app_dir_path) == str:
            raise TypeError 
        if not type(self.infer_run_mode) == list:
            raise TypeError 
        if not type(self.human_measure) == str:
            raise TypeError 
        if not type(self.base_metric) == str:
            raise TypeError
        if not type(self.gt_weightmap_types) == list:
            raise TypeError 
        if not type(self.click_weightmaps_dict) == dict:
            raise TypeError
        if not type(self.infer_run_parametrisation) == dict:
            raise TypeError
        if not type(self.infer_run_nums) == list:
            raise TypeError 
        if not type(self.infer_simulation_type) == str:
            raise TypeError
        if not type(self.checkpoint) == str:
            raise TypeError 
        if not type(self.datetime) == str:
            raise TypeError
        if not type(self.studies) == str:
            raise TypeError 
        if not type(self.summary_dict) == dict:
            raise TypeError 
        
        # Checking that the inference mode is an editing mode, because otherwise there is no need for a temporal component to the plotter.
        if self.infer_run_mode[0] != 'Editing':
            raise ValueError('The trend plotter is only intended for inference runs with > 1 iteration of inference.')

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
                                     'Maximum', 
                                     'Mean Relative Improvement to Init',
                                     'Median Relative Improvement to Init',
                                     'Standard Deviation of Relative Improvement to Init',
                                     'Interquartile Range of Relative Improvement to Init',
                                     'Lower Quartile of Relative Improvement to Init',
                                     'Upper Quartile of Relative Improvement to Init',
                                     'Minimum of Relative Improvement to Init',
                                     'Maximum of Relative Improvement to Init',
                                     'Mean Per Iter Improvement',
                                     'Median Per Iter Improvement',
                                     'Standard Deviation of Per Iter Improvement',
                                     'Interquartile Range of Per Iter Improvement',
                                     'Lower Quartile of Per Iter Improvement',
                                     'Upper Quartile of Per Iter Improvement',
                                     'Minimum of Per Iter Improvement',
                                     'Maximum of Per Iter Improvement']
        
        supported_simulation_types = ['probabilistic',
                                      'deterministic']
        

        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_score_summaries, supported_simulation_types
          

    def score_extraction(self, results_save_dir, metric):
        
        assert type(results_save_dir) == str 
        assert os.path.exists(results_save_dir) 

        #obtaining the paths for all of the score files we want to merge together:
        score_path = os.path.join(results_save_dir, f'{metric}_per_sample_averaged_results.csv')

        assert os.path.exists(score_path) 

        # num_experiment_repeats = len(self.infer_run_nums) 

        # valid_sample_indices = [j for sublist in [list(range(self.total_samples * i,  self.total_samples * i + self.num_samples)) for i in range(num_experiment_repeats)] for j in sublist]


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
        
        return scores


    def __call__(self):
    
        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode

        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset + f'_{self.infer_simulation_type}'
        
        inference_config_dict['datetime'] = self.datetime
        inference_config_dict['checkpoint'] = self.checkpoint

        inference_config_dict['inference_click_parametrisation'] = self.infer_run_parametrisation 

        if not type(self.infer_run_nums) == list:
            raise TypeError('Inference run nums must be presented as a list of strings')

        inference_config_dict['run_infer_string'] = 'run' + "".join([f"_{run}" for run in self.infer_run_nums])
        
        metric_config_dict = dict() 
        metric_config_dict['click_weightmap_types'] = list(self.click_weightmaps_dict.keys())
        
        #The click parametrisations dict is re-assigned, which we will later save alongside the computed metrics in their corresponding folders for reference later.    
        metric_config_dict['click_weightmap_parametrisations'] = self.click_weightmaps_dict 
        

        #Verifying that the selected configurations are supported by the downstream utilities.

        supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_score_summaries, supported_simulation_types = self.supported_configs()


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
        
        if any([summary not in supported_score_summaries for summary in self.summary_dict.keys()]): 
            raise ValueError("The selected score summaries are not yet supported")
        
        if self.infer_simulation_type not in supported_simulation_types:

            raise ValueError("The selected simulation type (e.g. probabilistic) was not supported")

        metric_config_dict['gt_weightmap_types'] = self.gt_weightmap_types
        metric_config_dict['human_measure'] = self.human_measure 
        metric_config_dict['base_metric'] = self.base_metric 

        ################################################################################################################################################################################

        #Generation of the paths required for extracting the (segmentations, guidance point sets, guidance point parametrisations etc.) and the path for saving the results.

        path_generation_class = path_generation(inference_config_dict, metric_config_dict)

        #Extracts the upper level location where the raw results, summary results etc folders will be placed.
        _, results_save_dir = path_generation_class()
        
        #We extract the dictionary of class-label - class-code correspondences. This should be located in the upper folder for the dataset at hand.

        label_config_path = os.path.join(self.app_dir_path, 'datasets', self.studies, 'label_configs.txt')
        
        assert os.path.exists(label_config_path)
        ################### Importing the class label configs dictionary #####################

        with open(label_config_path) as f:
            class_config_dict = json.load(f)

        config_labels = class_config_dict["labels"]

        results_summarisation_path = os.path.join(results_save_dir, 'results_summarisation', f'{self.base_metric}_summarisation.csv')
        
        assert os.path.exists(results_summarisation_dir)

        extracted_summarisation_scores = self.score_extraction(self.base_metric)

        # filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)

        self.score_summarisation(results_summarisation_dir, f'{self.base_metric}_summarisation.csv', extracted_scores)

        if self.per_class_scores:

            for class_label in config_labels.keys():
                
                if not self.include_background_metric:
                    if class_label.title() == "Background":
                        continue 
                
                extracted_scores = self.score_extraction(os.path.join(results_save_dir, 'per_sample_averaged_results'), f'class_{class_label}_{self.base_metric}')
                
                # filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)
                
                self.score_summarisation(results_summarisation_dir, f'class_{class_label}_{self.base_metric}_summarisation.csv', extracted_scores)