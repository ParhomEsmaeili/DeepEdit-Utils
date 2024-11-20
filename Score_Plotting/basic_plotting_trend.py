import numpy as np
import matplotlib.pyplot as plt
import os
import json 
# from pathlib import Path 
# import torch 
import numpy as np
import csv
import itertools
from os.path import dirname as up
import sys 

utils_dir = up(up(up(os.path.abspath(__file__))))
sys.path.append(utils_dir)

from Metric_Computation_Utils.score_generation_path_utils import path_generation 
from Miscellaneous.graph_structure_generator import Graph 
# import shutil 
# import math 


'''Script which is used to generate a plotted trend against iterations for a given run/metric and summarisation set of summarisation statistics'''


class plot_trend:
    
    def __init__(self, args):
        
        if not type(args) == dict:
            raise TypeError('Score generation failed because the score merging config was not a dict')

        self.dataset_subset = args['dataset_subset']
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
        self.checkpoints = args['checkpoints']
        self.datetimes = args['datetimes']
        self.studies = args['studies'] 
        self.summary_dict = args['summary_dict']
        self.plot_info = args['plot_info'] 
        #Plot_info is a dictionary which contains any information regarding the plot, which cannot be extracted from the information already provided:
        #E.g. Legend, Whether the time-dimension values should be generating a plot that is smoothed and the smoothing strategy.. 

        if not type(self.dataset_subset) == str:
            raise TypeError  
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
        if not type(self.checkpoints) == str:
            raise TypeError 
        if not type(self.datetimes) == str:
            raise TypeError
        if not type(self.studies) == str:
            raise TypeError 
        if not type(self.summary_dict) == dict:
            raise TypeError 
        if not type(self.plot_info) == dict:
            raise TypeError 
        # Checking that the inference mode is an editing mode, because otherwise there is no need for a temporal component to a plotter.
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
                                     ]
        
        supported_simulation_types = ['probabilistic',
                                      'deterministic']
    
        #We represent the permitted and forbidden combinations as a ConnectedGraph, if there is no direct edge between nodes (statistics) then it is forbidden.

        connections = [('Median', 'Lower Quartile'),
                        ('Median', 'Upper Quartile'),
                        ('Median', 'Minimum'),
                        ('Median', 'Maximum'), 
                        ('Median', 'Interquartile Range'), 
                        ('Mean','Standard Deviation'),
                        ('Mean', 'Minimum'),
                        ('Mean', 'Maximum')]
        
        SupportedCombinationsGraph = Graph(connections, directed=False)
        
        #We assert that all of the summarisation scores being used must have a connected edge to one another otherwise it is invalid.
        for combination in list(itertools.combinations(self.summary_dict.keys(),2)):
            if not SupportedCombinationsGraph.is_connected(combination[0], combination[1]):
                raise ValueError('The combination of these summary scores is not supported and/or permitted.')
          
        return (supported_initialisations, 
                supported_click_weightmaps, 
                supported_gt_weightmaps, 
                supported_human_measures, 
                supported_base_metrics, 
                supported_score_summaries, 
                supported_simulation_types)
          
    def generate_results_dirs(self):
        
        results_save_dirs = dict()

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

        for i in range(len(self.datetimes)):
            
            inference_config_dict['datetime'] = self.datetimes[i] 
            inference_config_dict['checkpoint'] = self.checkpoints[i] 

            results_save_dirs[self.datetimes[i]] = [self.checkpoints[i], os.path.join(path_generation(inference_config_dict, metric_config_dict)()[1], 'results_summarisation')]

        
        return results_save_dirs
    
    def score_extraction(self, summarisation_paths: dict[str, list[str]]):
        
        #extracting the scores
        summary_scores_dict = dict() 
        for datetime, [checkpoint, path] in summarisation_paths.items():
            #Initialising a dictionary of scores for each summary statistic for each model.
            summary_scores_dict[datetime] = {'checkpoint':checkpoint} 

            with open(path, newline='') as f:
                score_reader = csv.reader(f, delimiter=' ', quotechar='|')
                for row in score_reader:
                    
                    if row == []:
                        #Empty row
                        continue 
                    else:
                        row_str_list = row[0].split(',')

                        if row_str_list[0] in self.summary_dict.keys():
                            #If summarisation score in the desired sets of plotters.
                            summary_score_name = row_str_list[0]
                            summary_scores_dict[datetime][summary_score_name] = [float(string) for string in row_str_list[1:]]
                                
            
        return summary_scores_dict


    def plot(self, summarisation_scores_dict: dict[str, dict[str, list]]):
        '''
        Args: summarisation_scores_dict: The dictionary for each subdict for each datetime (model version) which contains the summarisation scores and the checkpoint used.
        '''
        
        if 'Median' in self.summary_dict.keys():
            
            if len(list(self.summary_dict.keys())) == 1:
                pass 
                #In this case, only median is being plotted. 
            
            elif len(list(self.summary_dict.keys())) != 1:
                #In this case, we are plotting the median, alongside other info..
                if 'Interquartile Range' in self.summary_dict.keys(): #Plot error bar
                    pass
                elif 'Lower Quartile' in self.summary_dict.keys() and 'Upper Quartile' in self.summary_dict.keys(): #Plot error bar
                    pass 
                
                elif 'Minimum' in self.summary_dict.keys() and 'Maximum' in self.summary_dict.keys():
                    pass #Plotting the minimum and maximum points also for the metric.
        
        elif 'Mean' in self.summary_dict.keys():

            if len(list(self.summary_dict.keys())) == 1:
                pass #In this case only the means are being plotted.
            
            elif len(list(self.summary_dict.keys())) > 1:
                #In this case, then we are plotting the mean, alongside other info.
                if 'Standard Deviation' in self.summary_dict.keys(): #Plot error bar
                    pass

                elif 'Minimum' in self.summary_dict.keys() and 'Maximum' in self.summary_dict.keys():
                    pass #Plotting the minimum and maximum point for the metric also..
                
    def initialise_figure(self):
        plt.figure()

    
    def plot_average_and_spread(self):
        raise NotImplementedError

    def plot_average_only(self):
        raise NotImplementedError
    
    def plot_extreme_points(self):
        raise NotImplementedError
    
    def extract_plot_parameters(self):
        '''
        args: Self attributes, which contain info about the inference run.

        returns: The following plot attributes: Title, Axes names
        '''
        #Extracting the title:

        if self.base_metric == 'Dice':
            
        #Returns a dictionary extracted from the information about the metric/inference run etc, which is required for the plot generation.
        return 
    
    def __call__(self):
    
        #Verifying that the selected configurations are supported by the downstream utilities.

        (supported_initialisations, 
         supported_click_weightmaps, 
         supported_gt_weightmaps, 
         supported_human_measures, 
         supported_base_metrics, 
         supported_score_summaries, 
         supported_simulation_types) = self.supported_configs()


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

        

        ################################################################################################################################################################################

       

        #Extracts the upper level location where the raw results, summary results etc folders will be placed.
        results_summarisation_dirs = self.generate_results_dirs()

    
        results_summarisation_paths = [os.path.join(results_summarisation_dirs, f'{self.base_metric}_summarisation.csv')]
        
        for path in results_summarisation_paths:
            if not os.path.exists(path):
                raise ValueError('The path was not valid')

        summarisation_scores = self.score_extraction(results_summarisation_paths) 
        
        #This is the dictionary which contains the list of scores for each summary statistic desired for plotting (for a given datetime/checkpoint).


        #Plotting:
        #First we extract parameters that pertain to the plot information: E.g. the title, axes names, etc. Then we plot.
        plot_params = self.extract_plot_parameters()
        self.plot(summarisation_scores)

