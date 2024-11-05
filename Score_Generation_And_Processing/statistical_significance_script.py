
'''

Script which contains the class/methods required for computing tests of statistical significance. This will be capable of accepting whatever the results being compared are
and outputs the statistical test results in a json file in a folder named: Statistical significance (same inner path for the inner folder)

'''

import os 
import shutil 
import csv 
from os.path import dirname as up
import sys 
utils_codebase__dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_codebase__dir)

from Metric_Computation_Utils.score_generation_path_utils import path_generation 
import json 

class statistical_significance_assessment:

    def __init__(self, args):

        self.statistical_test = args['statistical_test'] # The statistical significance test that is being used. 
        self.dataset_subset = args['dataset_subset']
        self.ignore_empty = args['ignore_empty']
        self.per_class_scores = args['per_class_scores']
        self.app_dir_path = os.path.join(os.path.expanduser("~"), args['app_dir'])
        self.infer_run_mode = args['inference_run_mode']
        self.human_measure = args['human_measure']
        self.base_metric = args['base_metric']
        self.gt_weightmap_types = args['gt_weightmap_types']
        self.click_weightmaps_dict = args['click_weightmap_dict']
        self.infer_run_parametrisation = args['inference_run_parametrisation'] #This parametrisation pertains to both the click size but also to whether it is working in CIM/1-iter SIM type modes
        self.infer_run_nums = args['inference_run_num'] #The infer run number being used.
        self.infer_simulation_type = args['simulation_type']
        self.checkpoint = args['checkpoint']
        self.datetimes = args['datetimes']
        self.studies = args['studies'] 


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
        assert type(self.gt_weightmap_types) == list 
        assert type(self.click_weightmaps_dict) == dict
        assert type(self.infer_run_parametrisation) == dict
        assert type(self.infer_run_nums) == list 
        assert type(self.infer_simulation_type) == str
        assert type(self.checkpoint) == str 
        assert type(self.datetimes) == list 
        assert type(self.studies) == str 


        
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
        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_simulation_types
          
    def generate_config_dicts(self):
        
        results_save_dirs = [] 

        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode
        
        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset + f'_{self.infer_simulation_type}'
    
        inference_config_dict['checkpoint'] = self.checkpoint

        inference_config_dict['inference_click_parametrisation'] = self.infer_run_parametrisation 
        inference_config_dict['run_infer_string'] = f'run_{self.infer_run_nums}'
        
        metric_config_dict = dict() 
        metric_config_dict['click_weightmap_types'] = list(self.click_weightmaps_dict.keys())
        
        #The click parametrisations dict is re-assigned, which we will later save alongside the computed metrics in their corresponding folders for reference later.    
        metric_config_dict['click_weightmap_parametrisations'] = self.click_weightmaps_dict 
                    
        metric_config_dict['gt_weightmap_types'] = self.gt_weightmap_types
        metric_config_dict['human_measure'] = self.human_measure 
        metric_config_dict['base_metric'] = self.base_metric 

        for datetime in self.datetimes:
            inference_config_dict['datetime'] = datetime 

            results_save_dirs.append(path_generation(inference_config_dict, metric_config_dict)[1])

            

        return results_save_dirs

    def __call__(self):

        #Verifying that the selected configurations are supported by the downstream utilities.

        supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_sequentiality_modes, supported_simulation_types = self.supported_configs()


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
        
