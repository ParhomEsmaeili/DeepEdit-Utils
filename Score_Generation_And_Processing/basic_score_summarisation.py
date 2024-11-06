import os
import json 
from pathlib import Path 
import torch 
import numpy as np
import csv 
from os.path import dirname as up
import sys 

utils_dir = up(up(os.path.abspath(__file__)))
from Metric_Computation_Utils.score_generation_path_utils import path_generation 
import shutil 
import math 


'''Most basic score summarisation script, which just uses the existing scores to compute summary statistics'''


class score_summarisation():
    def __init__(self, args):
        
        assert type(args) == dict, 'Score generation failed because the score merging config was not a dict'


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
        self.summary_dict = args['summary_dict']


        assert type(self.dataset_subset) == str 
        # assert type(self.sequentiality_mode) == str 
        # assert type(self.ignore_empty) == bool 
        assert type(self.per_class_scores) == bool 
        # assert type(self.include_background_mask) == bool 
        assert type(self.include_background_metric) == bool 
        assert type(self.app_dir_path) == str 
        assert type(self.infer_run_mode) == list 
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
        assert type(self.summary_dict) == dict 

    def supported_configs(self): 

        supported_initialisations = ['Autoseg','Interactive']
        
        supported_click_weightmaps =['Ellipsoid',
                                    'Cuboid', 
                                    'Scaled Euclidean Distance',
                                    'Exponentialised Scaled Euclidean Distance',
                                    '2D Intersections', 
                                    'None']
        
        supported_gt_weightmaps = ['Connected Component',
                                    'None']
            
        supported_human_measures = ['Local Responsiveness',
                                    'Temporal Non Worsening',
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

        ellipsoid: raw parametrisations for each of the denoms in the ellipse equation, accepts 1 or N_dims length list
        cuboid: raw parametrisations for the half-dimension lengths in the cuboids, accepts 1 or N_dims length list 
        scaled euclidean distance: parametrisations which scale the terms in the euclidean (their denominators), accepts 1 or N_dims length list
        exponentialised scaled euclidean distance: parameterisations which scale the terms in eucilidean (their denoms) + the exponentiation parameter, accepts length 2 or N_dims + 1 length list.
        2d intersecions: none
        none: none 

        gt_weightmaps:
        
        connected_component: none
        none: none 

        '''
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

    # def per_sample_averaging(self, scores):
        
    #     num_experiments = len(self.infer_run_nums)

    #     #we assume the image names are still there in the first index! 
    #     output = [scores[0]] 

    #     #We filter the nan scores when computing averages, if for each sample there is not a non-nan score then just continue..

    #     for sublist in scores[1:]:
            
    #         current_iter_averaged = [] 

    #         for index in range(self.num_samples):
    #             experiment_values = [sublist[j * self.num_samples + index] for j in range(num_experiments)]
                
    #             if not self.include_nan:
    #                 non_nan_vals = [val for val in experiment_values if not math.isnan(float(val))]
    #                 if len(non_nan_vals)  == 0:
    #                     #in this case just skip to the next sample 
    #                     continue 
    #                 else:
    #                     #in this case, we used the filtered out nan values and average.
    #                     per_sample_mean = np.mean(non_nan_vals)
    #                     current_iter_averaged.append(per_sample_mean)

    #         #We then append the per sample averaged scores for that iteration.    
    #         output.append(current_iter_averaged)

    #     return output 
    
    def score_summarisation(self, results_summarisation_dir, filename, scores):

        
        just_scores = scores[1:] #Nested list of per iteration scores.
        
        summarised_output = dict() #We will save the summaries into this dict, which we will then use to save the scores to a csv file.

        for key in self.summary_dict.keys():
            parametrisation = self.summary_dict[key] 

            if key.title() == 'Mean':

                summarised_output[key] = self.compute_mean(just_scores, parametrisation)

            elif key.title() == "Median":

                summarised_output[key] = self.compute_median(just_scores, parametrisation)

            elif key.title() == "Standard Deviation":

                summarised_output[key] = self.compute_standard_dev(just_scores, parametrisation)

            elif key.title() == "Interquartile Range":

                summarised_output[key] = self.compute_iqr(just_scores, parametrisation) 

            elif key.title() == "Lower Quartile":

                summarised_output[key] = self.compute_lower_quartile(just_scores, parametrisation)
            
            elif key.title() == "Upper Quartile":

                summarised_output[key] = self.compute_upper_quartile(just_scores, parametrisation)

            elif key.title() == "Minimum":

                summarised_output[key] = self.compute_minimum(just_scores, parametrisation)

            elif key.title() == "Maximum":
                
                summarised_output[key] = self.compute_maximum(just_scores, parametrisation)

            # elif key.title() == "Mean Relative Improvement":

            #     relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)  
            #     summarised_output[key] = self.compute_mean(relative_improv_scores)

            # elif key.title() == "Median Relative Improvement":
                
            #     relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)
            #     summarised_output[key] = self.compute_median(relative_improv_scores)




        

        #Saving the summary statistics
        
        with open(os.path.join(results_summarisation_dir, filename), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])

        for summary_statistic_key in summarised_output.keys():
            summary_stat_row = [summary_statistic_key]

            for score in summarised_output[summary_statistic_key]:
                summary_stat_row.append(score) 

            with open(os.path.join(results_summarisation_dir, filename),'a') as f:
                
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(summary_stat_row)   

    
    
    def compute_mean(self,output_scores, parametrisation): 
        #Any potential nans need to be filtered (which may arise for instances where no clicks are provided for a given class multiple times for a given sample for click based metric, very unlucky.) 
          
        return [np.mean([val for val in sublist if not math.isnan(val)]) for sublist in output_scores]
    
    def compute_median(self, output_scores, parametrisation):

        return [np.median([val for val in sublist if not math.isnan(val)]) for sublist in output_scores]

    def compute_standard_dev(self, output_scores, parametrisation): 

        return [np.std([val for val in sublist if not math.isnan(val)], dtype=np.float64) for sublist in output_scores]
    
    def compute_iqr(self, output_scores, parametrisation):

        return [np.percentile([val for val in sublist if not math.isnan(val)], 75) - np.percentile(sublist, 25) for sublist in output_scores]
    
    def compute_upper_quartile(self, output_scores, parametrisation):

        return [np.percentile([val for val in sublist if not math.isnan(val)], 75) for sublist in output_scores]
    
    def compute_lower_quartile(self, output_scores, parametrisation):

        return [np.percentile([val for val in sublist if not math.isnan(val)], 25) for sublist in output_scores]
    
    def compute_minimum(self, output_scores, parametrisation):

        return [np.min([val for val in sublist if not math.isnan(val)]) for sublist in output_scores]
    
    def compute_maximum(self, output_scores, parametrisation):

        return [np.max([val for val in sublist if not math.isnan(val)]) for sublist in output_scores]
    
    
    def compute_relative_improvement(self, output_scores, parametrisation):
        #This is relative to the initialisation score!
        #This requires the number of samples to be consistent throughout! 
        
        num_scores = len(output_scores[0])
        for score_list in output_scores:
            assert len(score_list) == num_scores, "There was an incongruence in the number of scores provided per iteration, for the relative to initialisation score improvement computation" 

        initialisation = np.array(output_scores[0])
        return [np.array([float('nan')] * num_scores)] + [np.array(sublist) - initialisation for sublist in output_scores[1:]]

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

        #Summarisation config should also be saved as a json for checking:
        
        results_summarisation_dir = os.path.join(results_save_dir, 'results_summarisation')
        
        os.makedirs(results_summarisation_dir, exist_ok=True)

        with open(os.path.join(results_summarisation_dir, 'summarisation_config.json'), 'w') as f:
            json.dump(dict(vars(self)), f)

        extracted_scores = self.score_extraction(os.path.join(results_save_dir, 'per_sample_averaged_results'), self.base_metric)

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