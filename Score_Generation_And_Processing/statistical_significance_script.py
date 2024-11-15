
'''

Script which contains the class/methods required for computing tests of statistical significance. This will be capable of accepting whatever the results being compared are
and outputs the statistical test results in a json file in a folder named: Statistical significance (same inner path for the inner folder)

It is only being implemented for the cross-class scores here..otherwise a lot of exception handling is required. 

It can accept any pairing of datetimes and checkpoints as long as every other configuration is matching (i.e. both must have the same sets of results, the same quantity
of samples in the set of results ETC, the same test set, the same inference simulation type etc.).

The per-class scores for the locality metric are just
intended as a surface level examination.. However, it may be desirable at a later point.

'''

import os 
import math
import shutil 
import csv 
import numpy as np
import json
from scipy.stats import wilcoxon 
from os.path import dirname as up
import sys 
utils_codebase__dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_codebase__dir)

from Metric_Computation_Utils.score_generation_path_utils import path_generation 


class statistical_significance_assessment:

    def __init__(self, args):

        self.statistical_tests = args['statistical_test'] # The statistical significance test that is being used. 
        self.dataset_subset = args['dataset_subset']
        # self.ignore_empty = args['ignore_empty']
        self.per_class_scores = args['per_class_scores']
        self.app_dir_path = os.path.join(os.path.expanduser("~"), args['app_dir'])
        self.infer_run_mode = args['inference_run_mode']
        self.human_measure = args['human_measure']
        self.base_metric = args['base_metric']
        self.derived_metrics = args['derived_metric']
        self.gt_weightmap_types = args['gt_weightmap_types']
        self.click_weightmaps_dict = args['click_weightmap_dict']
        self.infer_run_parametrisation = args['inference_run_parametrisation'] #This parametrisation pertains to both the click size but also to whether it is working in CIM/1-iter SIM type modes
        self.infer_run_nums = args['inference_run_nums'] #The infer run number being used.
        self.infer_simulation_type = args['simulation_type']
        self.checkpoints = args['checkpoints']
        self.datetimes = args['datetimes']
        self.studies = args['studies'] 
        # self.num_samples = args['num_samples']
        # self.total_samples = args['total_samples']

        assert type(self.dataset_subset) == str 
        # assert type(self.sequentiality_mode) == str 
        # assert type(self.ignore_empty) == bool 
        assert type(self.per_class_scores) == bool 
        # assert type(self.include_background_mask) == bool 
        # assert type(self.include_background_metric) == bool 
        assert type(self.app_dir_path) == str 
        assert type(self.infer_run_mode) == list 
        assert type(self.human_measure) == str 
        assert type(self.base_metric) == str
        assert type(self.derived_metrics) == list
        assert type(self.gt_weightmap_types) == list 
        assert type(self.click_weightmaps_dict) == dict
        assert type(self.infer_run_parametrisation) == dict
        assert type(self.infer_run_nums) == list 
        assert type(self.infer_simulation_type) == str
        assert type(self.checkpoints) == list 
        assert type(self.datetimes) == list 
        assert type(self.studies) == str
        assert type(self.statistical_tests) == dict
        # assert type(self.num_samples) == int 
        # assert type(self.total_samples) == int

        
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
        
        supported_simulation_types = ['probabilistic',
                                      'deterministic']
        
        supported_statistical_tests = ['Wilcoxon Signed Rank Test',
                                       'Paired T Test']
        
        supported_derived_metrics = ['Default',
                                    'Relative To Init Score',
                                    'Per Iter Improvement Score']
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
        return (supported_initialisations,
                supported_click_weightmaps, 
                supported_gt_weightmaps, 
                supported_human_measures, 
                supported_base_metrics, 
                supported_simulation_types, 
                supported_statistical_tests,
                supported_derived_metrics) 
          
    def generate_results_dirs(self):
        
        results_save_dirs = [] 

        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode
        
        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset + f'_{self.infer_simulation_type}'

        inference_config_dict['inference_click_parametrisation'] = self.infer_run_parametrisation 
        inference_config_dict['run_infer_string'] = 'run' + "".join([f"_{run}" for run in self.infer_run_nums])
        
        metric_config_dict = dict() 
        metric_config_dict['click_weightmap_types'] = list(self.click_weightmaps_dict.keys())
        
        #The click parametrisations dict is re-assigned, which we will later save alongside the computed metrics in their corresponding folders for reference later.    
        metric_config_dict['click_weightmap_parametrisations'] = self.click_weightmaps_dict 
                    
        metric_config_dict['gt_weightmap_types'] = self.gt_weightmap_types
        metric_config_dict['human_measure'] = self.human_measure 
        metric_config_dict['base_metric'] = self.base_metric 

        for i in range(2):
            
            inference_config_dict['datetime'] = self.datetimes[i] 
            inference_config_dict['checkpoint'] = self.checkpoints[i] 

            results_save_dirs.append(os.path.join(path_generation(inference_config_dict, metric_config_dict)()[1], 'per_sample_averaged_results'))

            

        return results_save_dirs

    def score_extraction(self, results_save_dirs, filename):
        
        set_of_scores = []
        for dir in results_save_dirs:
            
            assert type(dir) == str 
            assert os.path.exists(dir)

            #obtaining the paths for all of the score files we want to merge together:
            score_path = os.path.join(dir, filename)

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
                
                set_of_scores.append(scores)

        return set_of_scores

    def statistical_test_check(self, results_1, results_2):

        #Assuming the results come in a nested list, where each sublist is a list of results across the samples.

        statistical_test_results_dict = dict() 

        for key in self.statistical_tests.keys():
            
            if key.title() == 'Wilcoxon Signed Rank Test':
                parametrisation = self.statistical_tests[key]
                statistical_test_results_dict[key] = self.wilcoxon_calc(results_1[1:], results_2[1:], parametrisation)  #We just want to compute it on the scores, not the image name


            elif key.title() == 'Student T Test':
                ValueError("Not yet implemented")
        

        return statistical_test_results_dict
    
    def wilcoxon_calc(self, results_1, results_2, parametrisation): 
        
        assert len(results_1) == len(results_2)
        
        for i in range(len(results_1)):
            assert len(results_1[i]) == len(results_2[i]), f'Number of samples for the given iteration {i} were not equivalent'
        
        p_val = parametrisation['p_value']
        return [str(wilcoxon(results_1[i], results_2[i]).pvalue < p_val) for i in range(len(results_1))]

    
    def __call__(self):

        #Verifying that the selected configurations are supported by the downstream utilities.

        (supported_initialisations, 
        supported_click_weightmaps, 
        supported_gt_weightmaps, 
        supported_human_measures, 
        supported_base_metrics, 
        supported_simulation_types, 
        supported_statistical_tests,
        supported_derived_metrics) = self.supported_configs()


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
    
        if any([test not in supported_statistical_tests for test in self.statistical_tests.keys()]):
            raise ValueError("One of the chosen statistical tests are not currently supported")
        
        if any([metric not in supported_derived_metrics for metric in self.derived_metrics]):
            raise ValueError("One of the chosen derived metrics are not currently supported")
        
        sample_averaged_results_dirs = self.generate_results_dirs() 

        #We generate the output dirs by following the assumption that we follow the following path structure:
        #/home
        #/parhomesmaeili
        #/DeepEditPlusPlus Development
        #/DeepEditPlusPlus
        #/datasets
        #/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised  -- Datatset name
        #/validation_probabilistic_results  -- data subset and infer simulation type
        #/Dice -- base metric
        #/human_measure_None -- human measure
        #/click_weightmaps_None_{'None': ['None']} -- metric click weightmap type
        #/gt_weightmaps_None -- metric gt weightmap type
        #/20241103_142602 -- datetime
        #/best_val_score_epoch --checkpoint/epoch
        #/Autoseg_initialisation_10_edit_iters -- infer run name
        #/No Click Param -- infer simulation click parametrisations
        #/run_0_1_2 -- inference run numbers string
        #/per_sample_averaged_results -- directory which the per-sampled averaged results belong in

        #We generate the output_dir by switching the datetime for a pair of datetimes, and we also switch the checkpoint name for a pair of checkpoints
        # e.g. instead of 10072024_201348 and 10072024_182402 we have: 10072024_201348 & 10072024_182402, and the same structure for the checkpoints. 

        #We also save the configuration used in our final output directory, so that there is no confusion about which checkpoint belongs to which model version. 
        
        #First we split the existing paths: 
        # For now the datetime occurs on the 13th index (12 in pythonic code), the checkpoint occurs on the 14th (13 in pythonic code indexing), the name of the 
        #infer test set and the click simulation type falls under index 8 (7 in pythonic code)

        path_structures_split = [sample_averaged_results_dirs[0].split('/'), sample_averaged_results_dirs[1].split('/')]
        dataset_subdir_index = 7
        model_version_index = 12 
        checkpoint_index = 13 

        #We zip these together and assert that all but the checkpoint and the datetime must be equivalent
        final_output_dir_path_list = []
        for index, (x,y) in enumerate(zip(path_structures_split[0], path_structures_split[1])):
            
            if index == model_version_index or index == checkpoint_index:
            
                final_output_dir_path_list.append(f'{x} & {y}')

            else:
                assert x == y, 'Mistake in the configuration selected'
                
                
                if index == dataset_subdir_index:
                    final_output_dir_path_list.append(f'{x}_significance_tests')
                else:
                    final_output_dir_path_list.append(x)

        output_dir_path = "/" + os.path.join(*final_output_dir_path_list)
        
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path)
        #We need to save the config in the output_dir_path 


        with open(os.path.join(output_dir_path, 'statistical_sig_test_config.json'), 'w') as f:
    
            json.dump(dict(vars(self)), f, indent=2)
        

        if not self.per_class_scores:
            
            for derived_metric in self.derived_metrics:
                
                if derived_metric.title() == 'Default':
                    results_1, results_2 = self.score_extraction(sample_averaged_results_dirs, f'{self.base_metric}_per_sample_averaged_results.csv')
                    output_test_dict = self.statistical_test_check(results_1, results_2)

                    save_path = os.path.join(output_dir_path, f'{derived_metric}.json')
                    with open(save_path, 'w') as f:
                        json.dump(output_test_dict, f, indent=2)

                elif derived_metric.title() == 'Relative To Init Score':
                    results_1, results_2 = self.score_extraction(sample_averaged_results_dirs, f'{self.base_metric}_per_sample_averaged_relative_to_init.csv' )
                    output_test_dict = self.statistical_test_check(results_1, results_2)

                    save_path = os.path.join(output_dir_path, f'{derived_metric}.json')
                    with open(save_path, 'w') as f:
                        json.dump(output_test_dict, f, indent=2)
                    
                elif derived_metric.title() == 'Per Iter Improvement Score':
                    results_1, results_2 == self.score_extraction(sample_averaged_results_dirs, f'{self.base_metric}_per_sample_averaged_per_iter.csv')
                    output_test_dict = self.statistical_test_check(results_1, results_2)

                    save_path = os.path.join(output_dir_path, f'{derived_metric}.json')
                    with open(save_path, 'w') as f:
                        json.dump(output_test_dict, f, indent=2)