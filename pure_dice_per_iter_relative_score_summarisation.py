import os
import json 
from pathlib import Path 
import torch 
import numpy as np
import csv 
from score_generation_path_utils import path_generation 
import shutil 
import math 


'''Pure dice score summarisation which uses the existing scores to compute summary statistics for relative improvement on a per-iteration basis.'''


class pure_dice_relative_score_summarisation():
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
        self.checkpoint = args['checkpoint']
        self.datetime = args['datetime']
        self.studies = args['studies'] 
        self.include_nan = args['include_nan']
        self.num_samples = args['num_samples']
        self.total_samples = args['total_samples']
        self.summary_dict = args['summary_dict']


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
        assert type(self.checkpoint) == str 
        assert type(self.datetime) == str
        assert type(self.studies) == str 
        assert type(self.include_nan) == bool
        assert type(self.summary_dict) == dict 

    def supported_configs(self): 

        supported_initialisations = ['Autoseg','Interactive']
        
        supported_click_weightmaps = ['None']
        
        supported_gt_weightmaps = ['None']
            
        supported_human_measures = ['None']

        supported_base_metrics = ['Dice']
        
        supported_score_summaries = ['Mean Relative Improvement to Init',
                                     'Median Relative Improvement to Init',
                                     'Standard Deviation of Relative Improvement to Init',
                                     'Interquartile Range of Relative Improvement to Init',
                                     'Mean Per Iter Improvement',
                                     'Median Per Iter Improvement',
                                     'Standard Deviation of Per Iter Improvement',
                                     'Interquartile Range of Per Iter Improvement',
                                     ]
        
        '''
        Corresponding parametrisations:

        click weightmaps:

        none: none 

        gt_weightmaps:
        
        none: none 

        '''
        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_score_summaries
          

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
                        #in this case just skip to the next sample 
                        continue 
                    else:
                        #in this case, we used the filtered out nan values and average.
                        per_sample_mean = np.mean(non_nan_vals)
                        current_iter_averaged.append(per_sample_mean)

            #We then append the per sample averaged scores for that iteration.    
            output.append(current_iter_averaged)

        return output 
    def score_summarisation(self, results_dir, filename, scores):
        
        just_scores = scores[1:] #Nested list of per iteration scores.
        
        summarised_output = dict() #We will save the summaries into this dict, which we will then use to save the scores to a csv file.

        for key in self.summary_dict.keys():
            parametrisation = self.summary_dict[key] 

            # if key.title() == 'Mean':

            #     summarised_output[key] = self.compute_mean(just_scores, parametrisation)

            # elif key.title() == "Median":

            #     summarised_output[key] = self.compute_median(just_scores, parametrisation)

            if key.title() == "Standard Deviation of Relative Improvement To Init":

                relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_standard_dev(relative_improv_scores, parametrisation)

            elif key.title() == "Interquartile Range of Relative Improvement To Init":
                
                relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_iqr(relative_improv_scores, parametrisation) 

            elif key.title() == "Mean Relative Improvement To Init":

                relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)  
                summarised_output[key] = self.compute_mean(relative_improv_scores, parametrisation)

            elif key.title() == "Median Relative Improvement To Init":
                
                relative_improv_scores = self.compute_relative_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_median(relative_improv_scores, parametrisation)

######################################################################################
            elif key.title() == "Standard Deviation Of Per Iter Improvement":

                relative_improv_scores = self.compute_per_iter_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_standard_dev(relative_improv_scores, parametrisation)

            elif key.title() == "Interquartile Range Of Per Iter Improvement":
                
                relative_improv_scores = self.compute_per_iter_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_iqr(relative_improv_scores, parametrisation) 

            elif key.title() == "Mean Per Iter Improvement":

                relative_improv_scores = self.compute_per_iter_improvement(just_scores, parametrisation)  
                summarised_output[key] = self.compute_mean(relative_improv_scores, parametrisation)

            elif key.title() == "Median Per Iter Improvement":
                
                relative_improv_scores = self.compute_per_iter_improvement(just_scores, parametrisation)
                summarised_output[key] = self.compute_median(relative_improv_scores, parametrisation)




        #Saving the summary statistics
        
        with open(os.path.join(results_dir, filename), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])

        for summary_statistic_key in summarised_output.keys():
            summary_stat_row = [summary_statistic_key]

            for score in summarised_output[summary_statistic_key]:
                summary_stat_row.append(score) 

            with open(os.path.join(results_dir, filename),'a') as f:
                
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(summary_stat_row)   

    
    
    def compute_mean(self,output_scores, parametrisation):    
        return [np.mean(sublist) for sublist in output_scores]
    
    def compute_median(self, output_scores, parametrisation):

        return [np.median(sublist) for sublist in output_scores]

    def compute_standard_dev(self, output_scores, parametrisation): 

        return [np.std(sublist, dtype=np.float64) for sublist in output_scores]
    
    def compute_iqr(self, output_scores, parametrisation):

        return [np.percentile(sublist, 75) - np.percentile(sublist, 25) for sublist in output_scores]
    
    def compute_relative_improvement(self, output_scores, parametrisation):
        #This is relative to the initialisation score!
        #This requires the number of samples to be consistent throughout! 
        
        num_scores = len(output_scores[0])
        for score_list in output_scores:
            assert len(score_list) == num_scores, "There was an incongruence in the number of scores provided per iteration, for the relative to initialisation score improvement computation" 

        initialisation = np.array(output_scores[0])
        return [np.array([float('nan')] * num_scores)] + [np.array(sublist) - initialisation for sublist in output_scores[1:]]

    def compute_per_iter_improvement(self, output_scores, parametrisation):
        #This is all relative to the prior iteration's dice score. 

        num_scores = len(output_scores[0])
        
        for score_list in output_scores:
            assert len(score_list) == num_scores, "There was an incongruence in the number of scores provided per iteration, for the per-iteration relative improvement score generation"

        return [np.array([float('nan')] * num_scores)] + [np.array(sublist) - output_scores[i] for i,sublist in enumerate(output_scores[1:])]

    def __call__(self):
    
        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode
        
        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset
        
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

        supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_score_summaries = self.supported_configs()


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
        

        with open(os.path.join(results_save_dir, 'relative_score_summarisation_config.json'), 'w') as f:
    
            json.dump(dict(vars(self)), f)

        extracted_scores = self.score_extraction(results_save_dir, self.base_metric)

        filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)

        self.score_summarisation(results_save_dir, f'{self.base_metric}_relative_score_summarisation.csv', filtered_and_sample_averaged)

        if self.per_class_scores:

            for class_label in config_labels.keys():
                
                if not self.include_background_metric:
                    if class_label.title() == "Background":
                        continue 
                
                extracted_scores = self.score_extraction(results_save_dir, f'class_{class_label}_{self.base_metric}')
                
                filtered_and_sample_averaged = self.per_sample_averaging(extracted_scores)
                
                self.score_summarisation(results_save_dir, f'class_{class_label}_{self.base_metric}_relative_score_summarisation.csv', filtered_and_sample_averaged)