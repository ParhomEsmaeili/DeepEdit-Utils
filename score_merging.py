import argparse 
import os 
import shutil 
import csv 
from score_generation_path_utils import path_generation 
import json 

class score_merging_class():
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
        assert type(self.checkpoint) == str 
        assert type(self.datetime) == str
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
        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics
          

    def score_collection(self, results_save_dir, infer_run_nums, score_files_base_dir, metric, rejection_value=0):
        #obtaining the paths for all of the score files we want to merge together:
        score_paths = [os.path.join(score_files_base_dir, f'run_{run_num}') for run_num in infer_run_nums]

        #extracting the scores and collecting them together.
        all_scores = []

        for path in score_paths:
            with open(os.path.join(path, f'{metric}_score_results.csv'), newline='') as f:
                score_reader = csv.reader(f, delimiter=' ', quotechar='|')
                first_row = f.readline()
                first_row = first_row.strip()
                #print(first_row)
                #n_cols = first_row.count(',') + 1 
                scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
                #print(dice_scores)
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

            for output_index in range(len(final_output_scores)):
                
                final_output_scores[output_index] += score_set[output_index] 


        non_rejected_rows = []
        for score_row_index in range(len(final_output_scores[0])): #final_output_dice_scores[1:]:

            dice_score_row = [final_output_scores[j][score_row_index] for j in range(len(final_output_scores))]
            if all(dice_score_row[1:]) >= rejection_value:
            
            # accepted_dice_scores = [i for i in dice_score_column if i >= rejection_value]
                non_rejected_rows.append(dice_score_row[1:])
        
        
        with open(os.path.join(results_save_dir, f'{metric}_score_results.csv'),'a') as f:
            writer = csv.writer(f)
            
            for i in range(len(final_output_scores[0])):
                output_row = [sublist[i] for sublist in final_output_scores]
                writer.writerow(output_row)

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

        supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics = self.supported_configs()


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
        
        # if self.sequentiality_mode not in supported_sequentiality_modes:
        #     raise ValueError("The selected sequentiality mode (e.g. CIM) was not supported")
        
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


        #If an existing set of results exists for the result directory then it should be deleted! We are not rewriting scores.
        if os.path.exists(results_save_dir) == True:
            shutil.rmtree(results_save_dir)
        os.makedirs(results_save_dir)

        #Extracting the base directory that each of the folders containing scores exists within:
        results_base_dir = os.path.dirname(results_save_dir)

        self.score_collection(results_save_dir, self.infer_run_nums, results_base_dir, self.base_metric, 0)

        if self.per_class_scores:

            for class_label in config_labels.keys():
                
                if not self.include_background_metric:
                    if class_label.title() == "Background":
                        continue 
                
                self.score_collection(results_save_dir, self.infer_run_nums, results_base_dir, f'class_{class_label}_{self.base_metric}', 0)