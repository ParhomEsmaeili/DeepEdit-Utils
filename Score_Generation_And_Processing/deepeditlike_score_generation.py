# import argparse 
import os
from os.path import dirname as up
import csv 
import json 
import sys 
import shutil 
import copy
import torch 
import re 
utils_codebase__dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_codebase__dir)

from Metric_Computation_Utils.base_human_centric_metric_computation import score_tool
from Metric_Computation_Utils.score_generation_path_utils import path_generation
import Metric_Computation_Utils.guidance_points_utils as guide_utils 
from itertools import chain

class test_scores():
    def __init__(self, args):

        assert type(args) == dict, 'Score generation failed because the score generation config was not a dict'


        #TODO: ADD assertions for each of the fields in the inputs correspondingly. 
        self.dataset_subset = args['dataset_subset']
        self.sequentiality_mode = args['sequentiality_mode']
        self.ignore_empty = args['ignore_empty']
        self.per_class_scores = args['per_class_scores']
        self.include_background_mask = args['include_background_mask']
        self.include_background_metric = args['include_background_metric']
        self.app_dir_path = os.path.join(os.path.expanduser("~"), args['app_dir'])
        self.infer_run_mode = args['inference_run_mode']
        self.human_measure = args['human_measure']
        self.base_metric = args['base_metric']
        self.gt_weightmap_types = args['gt_weightmap_types']
        self.click_weightmaps_dict = args['click_weightmap_dict']
        self.infer_run_parametrisation = args['inference_run_parametrisation'] #This parametrisation pertains to both the click size but also to whether it is working in CIM/1-iter SIM type modes
        self.infer_run_num = args['inference_run_num']
        self.infer_simulation_type = args['simulation_type']
        self.checkpoint = args['checkpoint']
        self.datetime = args['datetime']
        self.studies = args['studies'] 

        assert type(self.dataset_subset) == str 
        assert type(self.sequentiality_mode) == str 
        assert type(self.ignore_empty) == bool 
        assert type(self.per_class_scores) == bool 
        assert type(self.include_background_mask) == bool 
        assert type(self.include_background_metric) == bool 
        assert type(self.app_dir_path) == str 
        assert type(self.infer_run_mode) == list 
        assert type(self.human_measure) == str 
        assert type(self.base_metric) == str
        assert type(self.gt_weightmap_types) == list 
        assert type(self.click_weightmaps_dict) == dict
        assert type(self.infer_run_parametrisation) == dict
        assert type(self.infer_run_num) == str 
        assert type(self.infer_simulation_type) == str
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
        
        supported_sequentiality_modes = ['CIM',
                                         'SIM']
        
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
        return supported_initialisations, supported_click_weightmaps, supported_gt_weightmaps, supported_human_measures, supported_base_metrics, supported_sequentiality_modes, supported_simulation_types
                                

    def base_score_computation(self,
                               scoring_tools, 
                               infer_run_mode, 
                               img_directory_path, 
                               results_save_dir, 
                               weightmap_parametrisations, 
                               human_measure, 
                               base_metric, 
                               per_class_scores, 
                               include_background_metric, 
                               config_labels, 
                               sequentiality_mode):
        '''
        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (SIM/only the clicks for the current iter.)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 1 or len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(include_background_metric) == bool 
        assert type(per_class_scores) == bool 
        assert type(config_labels) == dict 
        
        #Implement the score computation scripts here.

        #Here we will generate the paths for the images with which we want to compute scores.
        
        #Obtaining list of image names, not done in numeric order:
        image_names = [x for x in os.listdir(img_directory_path) if x.endswith('.nii.gz')]
        gt_image_folder = os.path.join(img_directory_path, 'labels', 'original')
        final_image_folder = os.path.join(img_directory_path,'labels', 'final')
        guidance_json_folder = os.path.join(img_directory_path,'labels','guidance_points')
        
        assert sequentiality_mode in ['SIM'], "The sequentiality mode was not supported for metric computation. It should only be for the SIM (1 iter assumed)" 


        assert human_measure.title() in ["None", "Local Responsiveness"], "The human measure did not match the supported ones"

        num_points = [] #Just checking that the number of points is roughly equal across SIM and CIM modes.

        if infer_run_mode[0].title() == "Editing":

            editing_score_tool = scoring_tools['Editing']
            initialisation_score_tool = scoring_tools[infer_run_mode[1].title()]
        

            initialisation_folder = os.path.join(img_directory_path, 'labels', infer_run_mode[1].lower())
            
            iteration_folders = [x for x in os.listdir(os.path.join(img_directory_path, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.

            for image in image_names:

                #Initialising the number of points list for the given image
                image_num_points = []

                #Extracting the image name without the file extension:
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)

                
                iter_infos = [[infer_run_mode[1].lower(), 'dummy']] #we use a dummy for the iter_name because it is assumed that each sublist is length 2: iter_type, iter_name 
                guidance_points_dict_init, guidance_points_init_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict_init.values() )))))



                #For the generation of the score we do assume that the image has the file type extension included! 
                
                cross_class_score_init, per_class_scores_init = initialisation_score_tool([initialisation_folder], gt_image_folder, image, guidance_points_dict_init, guidance_points_init_parametrisations) # Adding the initialisation score
                
                assert type(cross_class_score_init) == torch.Tensor
                assert type(per_class_scores_init) == dict 

                scores['cross_class'].append(float(cross_class_score_init))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_init[score_key]))
                
                ################################################################################

                #Adding the scores on the intermediary iterations

                
                for index,iteration_folder in enumerate(iteration_folders): 

                    #First we extract the guidance points and parametrisations for this iteration:

                    iter_infos = [['deepedit', index + 1]] 
                    guidance_points_dict, guidance_points_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                    #Appending the number of points
                    image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict.values() )))))

                    cross_class_score, per_class_scores_dict = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)

                    assert type(cross_class_score) == torch.Tensor
                    assert type(per_class_scores_dict) == dict 

                    scores['cross_class'].append(float(cross_class_score))

                    for score_key in config_labels.keys():
                        
                        if not include_background_metric:
                            if score_key.title() == "Background":
                                continue
                        
                        scores[score_key].append(float(per_class_scores_dict[score_key]))
                
                
                #for the final image
                
                
                iter_infos = [['final', 'dummy']] 
                guidance_points_final, guidance_final_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)

                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_final.values() )))))
                num_points.append(image_num_points)

                cross_class_score_final, per_class_scores_dict_final = editing_score_tool([final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)

                assert type(cross_class_score_final) == torch.Tensor
                assert type(per_class_scores_dict_final) == dict 

                scores['cross_class'].append(float(cross_class_score_final))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_dict_final[score_key]))


                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                            with open(os.path.join(results_save_dir, f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores[class_label])

                    with open(os.path.join(results_save_dir, 'raw_results', f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])
                else:
                    with open(os.path.join(results_save_dir, 'raw_results', f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

        else: #In this case, we just have the single iter of inference (an init mode)
            initialisation_score_tool = scoring_tools[infer_run_mode[0].title()]
            for image in image_names:

                #Initialising the number of points list for the given image
                image_num_points = []


                #Extracting the image name without the file extension:
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)

                
                iter_infos = [[infer_run_mode[0].lower(), 'dummy']] #we use a dummy for the iter_name because it is assumed that each sublist is length 2: iter_type, iter_name 
                guidance_points_dict_init, guidance_points_init_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict_init.values() )))))
                num_points.append(image_num_points)

                #For the generation of the score we do assume that the image has the file type extension included! 
                
                cross_class_score_init, per_class_scores_init = initialisation_score_tool([final_image_folder], gt_image_folder, image, guidance_points_dict_init, guidance_points_init_parametrisations) # Adding the initialisation score
                
                assert type(cross_class_score_init) == torch.Tensor
                assert type(per_class_scores_init) == dict 

                scores['cross_class'].append(float(cross_class_score_init))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_init[score_key]))

                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                            with open(os.path.join(results_save_dir, 'raw_results', f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores[class_label])

                    with open(os.path.join(results_save_dir, 'raw_results', f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

                else:
                    with open(os.path.join(results_save_dir, 'raw_results', f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

        #flatten and sum the total number of points! use the following function in the debugger to verify the number of points make sense (or manipulate the number of samples used)
        # def flatten(xss):
        #   return [x for xs in xss for x in xs]
        pass 
        

            
    def cim_mode_base_score_computation(self,
                                        scoring_tools, 
                                        infer_run_mode, 
                                        img_directory_path, 
                                        results_save_dir, 
                                        weightmap_parametrisations, 
                                        human_measure, 
                                        base_metric, 
                                        per_class_scores, 
                                        include_background_metric, 
                                        config_labels, 
                                        sequentiality_mode):
        '''
        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (CIM/clicks accumulated across iters)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 1 or len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(include_background_metric) == bool 
        assert type(per_class_scores) == bool 
        assert type(config_labels) == dict 
        
        #Implement the score computation scripts here.

        #Here we will generate the paths for the images with which we want to compute scores.
        
        #Obtaining list of image names, not done in numeric order:
        image_names = [x for x in os.listdir(img_directory_path) if x.endswith('.nii.gz')]
        gt_image_folder = os.path.join(img_directory_path, 'labels', 'original')
        final_image_folder = os.path.join(img_directory_path,'labels', 'final')
        guidance_json_folder = os.path.join(img_directory_path,'labels','guidance_points')
        
        assert sequentiality_mode in ['CIM'], "The sequentiality mode was not supported for metric computation. It should only be for the SIM (1 iter assumed) since the base metrics are irrespective of any guidance point info"
        
        assert human_measure.title() in ["None", "Local Responsiveness"], "The human measure did not match the supported ones"
        
        num_points = [] #Just checking that the number of points is roughly equal across SIM and CIM modes.

        if infer_run_mode[0].title() == "Editing":

            editing_score_tool = scoring_tools['Editing']
            initialisation_score_tool = scoring_tools[infer_run_mode[1].title()]
        

            initialisation_folder = os.path.join(img_directory_path, 'labels', infer_run_mode[1].lower())
            
            iteration_folders = [x for x in os.listdir(os.path.join(img_directory_path, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.

            for image in image_names:

                #Initialising the number of points list for the given image
                image_num_points = []

                #Extracting the image name without the file extension:
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)

                
                #For the initialisation, there is no prior iter. 

                iter_infos = [[infer_run_mode[1].lower(), 'dummy']] #we use a dummy for the iter_name because it is assumed that each sublist is length 2: iter_type, iter_name 
                
                guidance_points_dict_init, guidance_points_init_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict_init.values() )))))

                #For the generation of the score we do assume that the image has the file type extension included! 
                
                cross_class_score_init, per_class_scores_init = initialisation_score_tool([initialisation_folder], gt_image_folder, image, guidance_points_dict_init, guidance_points_init_parametrisations) # Adding the initialisation score
                
                assert type(cross_class_score_init) == torch.Tensor
                assert type(per_class_scores_init) == dict 

                scores['cross_class'].append(float(cross_class_score_init))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_init[score_key]))
                
                ################################################################################

                #Adding the scores on the intermediary iterations

                
                for index,iteration_folder in enumerate(iteration_folders): 

                    #First we extract the guidance points and parametrisations for this iteration:

                    iter_infos += [['deepedit', index + 1]] 
                    
                    submitted_iter_infos = iter_infos[-2:]

                    guidance_points_dict, guidance_points_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, submitted_iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                    #Appending the number of points
                    image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict.values() )))))


                    cross_class_score, per_class_scores_dict = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)

                    assert type(cross_class_score) == torch.Tensor
                    assert type(per_class_scores_dict) == dict 

                    scores['cross_class'].append(float(cross_class_score))

                    for score_key in config_labels.keys():
                        
                        if not include_background_metric:
                            if score_key.title() == "Background":
                                continue
                        
                        scores[score_key].append(float(per_class_scores_dict[score_key]))
                
                
                #for the final image
                
                
                iter_infos += [['final', 'dummy']]

                submitted_iter_infos = iter_infos[-2:] 

                guidance_points_final, guidance_final_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, submitted_iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
            
                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_final.values() )))))

                cross_class_score_final, per_class_scores_dict_final = editing_score_tool([final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)

                assert type(cross_class_score_final) == torch.Tensor
                assert type(per_class_scores_dict_final) == dict 

                scores['cross_class'].append(float(cross_class_score_final))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_dict_final[score_key]))


                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                            with open(os.path.join(results_save_dir, f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores[class_label])

                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])
                else:
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

                
                num_points.append(image_num_points)

        else: #In this case, we just have the single iter of inference (an init mode)
            initialisation_score_tool = scoring_tools[infer_run_mode[0].title()]
            for image in image_names:
                
                #Initialising the number of points list for the given image
                image_num_points = []

                #Extracting the image name without the file extension:
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)

                
                iter_infos = [[infer_run_mode[0].lower(), 'dummy']] #we use a dummy for the iter_name because it is assumed that each sublist is length 2: iter_type, iter_name 
                guidance_points_dict_init, guidance_points_init_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                #Appending the number of points
                image_num_points.append(len(list(chain.from_iterable(list(guidance_points_dict_init.values() )))))

                #For the generation of the score we do assume that the image has the file type extension included! 
                
                cross_class_score_init, per_class_scores_init = initialisation_score_tool([final_image_folder], gt_image_folder, image, guidance_points_dict_init, guidance_points_init_parametrisations) # Adding the initialisation score
                
                assert type(cross_class_score_init) == torch.Tensor
                assert type(per_class_scores_init) == dict 

                scores['cross_class'].append(float(cross_class_score_init))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_init[score_key]))

                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                            with open(os.path.join(results_save_dir, f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(scores[class_label])
                    
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

                else:
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

                num_points.append(image_num_points)
        
        #flatten and sum the total number of points! use the following function in the debugger to verify the number of points make sense (or manipulate the number of samples used)
        # def flatten(xss):
        #   return [x for xs in xss for x in xs]
        pass 

    def sim_temporal_computation(self,
                               scoring_tools, 
                               infer_run_mode, 
                               img_directory_path, 
                               results_save_dir, 
                               weightmap_parametrisations, 
                               human_measure, 
                               base_metric, 
                               per_class_scores, 
                               include_background_metric, 
                               config_labels, 
                               sequentiality_mode):
        '''
        This method is typically assumed to be used for temporal non-worsening, and as such the code downstream will find the changed voxels, and perform metrics by comparing those voxels to the 
        true ground truth.

        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points to use for weightmaps!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (SIM/only the clicks for the current iter.)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(include_background_metric) == bool 
        assert type(per_class_scores) == bool 
        assert type(config_labels) == dict 
        
        #Implement the score computation scripts here.

        #Here we will generate the paths for the images with which we want to compute scores.
        
        assert sequentiality_mode in ['SIM'], "The sequentiality mode was not supported for metric computation. It should only be for the SIM (1 iter assumed)"

        assert human_measure.title() in ["Local Responsiveness", "Temporal Non Worsening"], "The human measure did not match the supported ones"

        #For temporal consistency measurement scores, this only applies to inference runs with iterative refinement. NOT for methods with only one iteration.
        
        #Obtaining list of image names, not done in numeric order.

        image_names = [x for x in os.listdir(img_directory_path) if x.endswith('.nii.gz')]
        
        #We may potentially need the ground truth maps for our metrics after all! 
        gt_image_folder = os.path.join(img_directory_path, 'labels', 'original')
        final_image_folder = os.path.join(img_directory_path,'labels', 'final')
        guidance_json_folder = os.path.join(img_directory_path,'labels','guidance_points')


        
        if infer_run_mode[0].title() == "Editing":

            editing_score_tool = scoring_tools['Editing']
            # initialisation_score_tool = scoring_tools[infer_run_mode[1].title()]
        

            initialisation_folder = os.path.join(img_directory_path, 'labels', infer_run_mode[1].lower())
            
            iteration_folders = [x for x in os.listdir(os.path.join(img_directory_path, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.

            for image in image_names:
                #Extracting the image name without the file extension:
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)
                
                ################################################################################

                #Adding the scores on the intermediary iterations

                
                for index,iteration_folder in enumerate(iteration_folders): 

                    #First we extract the guidance points and parametrisations for this iteration:

                    iter_infos = [['deepedit', index + 1]] 
                    guidance_points_dict, guidance_points_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                

                    if index == 0:
                        #In this case it is the first editing iteration, and so we use the init folder, and the first iteration folder. 

                        #If temporal consistency score
                        if self.human_measure == 'Temporal Non Worsening':
                            cross_class_score, per_class_scores_dict = editing_score_tool([initialisation_folder, os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                        elif self.human_measure == "Local Responsiveness":
                            #In this case, we want to find the pre and post-click segmentation scores weighted by the mask for the given iteration.
                            cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([initialisation_folder], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score = cross_class_score_post_click - cross_class_score_pre_click 
                            
                            per_class_scores_dict = dict() 

                            for class_label in per_class_scores_dict_pre_click.keys():
                                per_class_scores_dict[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]

                    else:
                        #In this case it is a non-first editing iteration, iteration. So we just use the current and prior iteration folders. 
                        
                        #If temporal consistency score
                        if self.human_measure == 'Temporal Non Worsening':
                            cross_class_score, per_class_scores_dict = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[index - 1]), os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                        
                        elif self.human_measure == "Local Responsiveness":
                            #In this case we want to find the pre and post click segmentation scores weighted by the mask for the current editing iteration 
                            cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[index - 1])], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[index])], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score = cross_class_score_post_click - cross_class_score_pre_click 
                            
                            per_class_scores_dict = dict() 

                            for class_label in per_class_scores_dict_pre_click.keys():
                                per_class_scores_dict[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]

                    assert type(cross_class_score) == torch.Tensor
                    assert type(per_class_scores_dict) == dict 

                    scores['cross_class'].append(float(cross_class_score))

                    for score_key in config_labels.keys():
                        
                        if not include_background_metric:
                            if score_key.title() == "Background":
                                continue
                        
                        scores[score_key].append(float(per_class_scores_dict[score_key]))
                
                
                #for the final image
                
                
                iter_infos = [['final', 'dummy']] 

                
                guidance_points_final, guidance_final_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                if self.human_measure == "Temporal Non Worsening":
                    cross_class_score_final, per_class_scores_dict_final = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[-1]), final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)

                elif self.human_measure == "Local Responsiveness":
                    
                    #In this case we want to find the pre and post click segmentation scores weighted by the mask for the current editing iteration 
                    cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[-1])], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)
                    cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)
                    cross_class_score_final = cross_class_score_post_click - cross_class_score_pre_click 
                    
                    per_class_scores_dict_final = dict() 

                    for class_label in per_class_scores_dict_pre_click.keys():
                        per_class_scores_dict_final[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]

                

                assert type(cross_class_score_final) == torch.Tensor
                assert type(per_class_scores_dict_final) == dict 

                scores['cross_class'].append(float(cross_class_score_final))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_dict_final[score_key]))


                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                        with open(os.path.join(results_save_dir, f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(scores[class_label])

                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])
                else:
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])
        
    def cim_temporal_computation(self,
                                scoring_tools, 
                                infer_run_mode, 
                                img_directory_path, 
                                results_save_dir, 
                                weightmap_parametrisations, 
                                human_measure, 
                                base_metric, 
                                per_class_scores, 
                                include_background_metric, 
                                config_labels, 
                                sequentiality_mode):
        '''
        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (CIM/clicks accumulated across iters)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(include_background_metric) == bool 
        assert type(per_class_scores) == bool 
        assert type(config_labels) == dict 
        
        #Implement the score computation scripts here.

        #Here we will generate the paths for the images with which we want to compute scores.
        
        #Obtaining list of image names, not done in numeric order:
        image_names = [x for x in os.listdir(img_directory_path) if x.endswith('.nii.gz')]
        gt_image_folder = os.path.join(img_directory_path, 'labels', 'original')
        final_image_folder = os.path.join(img_directory_path,'labels', 'final')
        guidance_json_folder = os.path.join(img_directory_path,'labels','guidance_points')
        
        assert sequentiality_mode in ['CIM'], "The sequentiality mode was not supported for metric computation. It should only be for the SIM (1 iter assumed) since the base metrics are irrespective of any guidance point info"
        
        assert human_measure.title() in ["Local Responsiveness", "Temporal Non Worsening"], "The human measure did not match the supported ones"
        
        if infer_run_mode[0].title() == "Editing":

            editing_score_tool = scoring_tools['Editing']
            # initialisation_score_tool = scoring_tools[infer_run_mode[1].title()]
        

            initialisation_folder = os.path.join(img_directory_path, 'labels', infer_run_mode[1].lower())
            
            iteration_folders = [x for x in os.listdir(os.path.join(img_directory_path, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.

            for image in image_names:
                #Extracting the image name without the file extension:
                
                
                
                image_no_ext = image.split('.')[0] 

                #We use a dict to save the scores, because we may also want per-class scores to be saved.
                scores = dict()

                for key in ['cross_class'] + list(config_labels.keys()):
                    if not include_background_metric:
                        if key.title() == "Background":
                            continue 
            
                    scores[key] = [image_no_ext]
                

                #We need to extract the guidance points and the guidance point parametrisations for the initialisation.

                #Iteration_Infos: The nested list containing info about the iterations under consideration
                # e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)

                ################################################################################

                #Adding the scores on the intermediary iterations

                iter_infos = [[infer_run_mode[1].lower(), 'dummy']]
                
                for index,iteration_folder in enumerate(iteration_folders): 
                     
                    #First we extract the guidance points and parametrisations for this iteration:

                    iter_infos += [['deepedit', index + 1]] 
                    
                    submitted_iter_infos = iter_infos[-2:]

                    guidance_points_dict, guidance_points_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, submitted_iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
                
                    if index == 0:
                        #In this case the prior iter was the initialisation. 

                        if self.human_measure == "Temporal Non Worsening":

                            cross_class_score, per_class_scores_dict = editing_score_tool([initialisation_folder, os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                    
                        elif self.human_measure == "Local Responsiveness":
                            #In this case, we want to find the pre and post-click segmentation scores weighted by the mask for the given iteration.
                            cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([initialisation_folder], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score = cross_class_score_post_click - cross_class_score_pre_click 
                            
                            per_class_scores_dict = dict() 

                            for class_label in per_class_scores_dict_pre_click.keys():
                                per_class_scores_dict[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]

                    else:
                        
                        if self.human_measure == "Temporal Non Worsening":
                            #In this case the prior iter was an edit iter.
                            cross_class_score, per_class_scores_dict = editing_score_tool([os.path.join(img_directory_path, 'label', iteration_folders[-1]), os.path.join(img_directory_path, 'labels', iteration_folder)], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                        

                        elif self.human_measure == "Local Responsiveness":
                            #In this case we want to find the pre and post click segmentation scores weighted by the mask for the current editing iteration 
                            cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[index - 1])], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[index])], gt_image_folder, image, guidance_points_dict, guidance_points_parametrisations)
                            cross_class_score = cross_class_score_post_click - cross_class_score_pre_click 
                            
                            per_class_scores_dict = dict() 

                            for class_label in per_class_scores_dict_pre_click.keys():
                                per_class_scores_dict[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]
                            
                    assert type(cross_class_score) == torch.Tensor
                    assert type(per_class_scores_dict) == dict 

                    scores['cross_class'].append(float(cross_class_score))

                    for score_key in config_labels.keys():
                        
                        if not include_background_metric:
                            if score_key.title() == "Background":
                                continue
                        
                        scores[score_key].append(float(per_class_scores_dict[score_key]))
                
                
                #for the final image
                
                
                iter_infos += [['final', 'dummy']]

                submitted_iter_infos = iter_infos[-2:] 

                guidance_points_final, guidance_final_parametrisations = guide_utils.guidance_dict_info(guidance_json_folder, image_no_ext, submitted_iter_infos, weightmap_parametrisations, sequentiality_mode, config_labels)
            

                if self.human_measure == "Temporal Non Worsening":
    
                    cross_class_score_final, per_class_scores_dict_final = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[-1]), final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)
            
                elif self.human_measure == "Local Responsiveness":
                    
                    #In this case we want to find the pre and post click segmentation scores weighted by the mask for the current editing iteration 
                    cross_class_score_pre_click, per_class_scores_dict_pre_click = editing_score_tool([os.path.join(img_directory_path, 'labels', iteration_folders[-1])], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)
                    cross_class_score_post_click, per_class_scores_dict_post_click = editing_score_tool([final_image_folder], gt_image_folder, image, guidance_points_final, guidance_final_parametrisations)
                    cross_class_score_final = cross_class_score_post_click - cross_class_score_pre_click 
                    
                    per_class_scores_dict_final = dict() 

                    for class_label in per_class_scores_dict_pre_click.keys():
                        per_class_scores_dict_final[class_label] = per_class_scores_dict_post_click[class_label] - per_class_scores_dict_pre_click[class_label]


                assert type(cross_class_score_final) == torch.Tensor
                assert type(per_class_scores_dict_final) == dict 

                scores['cross_class'].append(float(cross_class_score_final))

                for score_key in config_labels.keys():
                    
                    if not include_background_metric:
                        if score_key.title() == "Background":
                            continue
                    
                    scores[score_key].append(float(per_class_scores_dict_final[score_key]))


                if per_class_scores:

                    for class_label in config_labels.keys():
                        
                        if not include_background_metric:
                            if class_label.title() == "Background":
                                continue 
                            
                        with open(os.path.join(results_save_dir, f'class_{class_label}_{base_metric}_score_results.csv'),'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(scores[class_label])
                    
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])

                else:
                    with open(os.path.join(results_save_dir, f'{base_metric}_score_results.csv'),'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores['cross_class'])


            
    def local_score_computation(self,
                                scoring_tools, 
                                infer_run_mode, 
                                img_directory_path, 
                                results_save_dir, 
                                weightmap_parametrisations, 
                                human_measure, 
                                base_metric, 
                                per_class_scores,
                                include_background_metric,
                                config_labels, 
                                sequentiality_mode):
        '''
        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (E.g. CIM/accumulated across iters, or SIM/only the clicks for the current iter.)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 1 or len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(per_class_scores) == bool 
        assert type(include_background_metric) == bool 
        assert type(config_labels) == dict 
    
        
        assert sequentiality_mode in ['CIM', 'SIM'], "The sequentiality mode was not supported for metric computation."

        assert human_measure.title() == "Local Responsiveness", "Human measure did not match the computation function"

        #In this case, the prediction, the ground truth, and the guidance points set, and the parametrisations for the weightmap for the metric are provided (only for a given iter). 

        if sequentiality_mode == "SIM":
            if len(infer_run_mode) == 1:
                self.base_score_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)
            if len(infer_run_mode) == 3:
                self.sim_temporal_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)
        if sequentiality_mode == "CIM":
            if len(infer_run_mode) == 1:
                self.cim_mode_base_score_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)
            if len(infer_run_mode) == 3:
                self.cim_temporal_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)
    

    def temporal_consist_score_computation(self,
                                           scoring_tools, 
                                           infer_run_mode, 
                                           img_directory_path, 
                                           results_save_dir, 
                                           weightmap_parametrisations, 
                                           human_measure, 
                                           base_metric, 
                                           per_class_scores,
                                           include_background_metric,
                                           config_labels, 
                                           sequentiality_mode):
        '''
        This method should compute the metric scores for the set of images and task which has been provided. It should 
        #save this in a csv file, the metric scores returned should provide the metric scores across all the images
        #that have been provided.
        # 
        Inputs: 
        The scoring tools which we will throughout the iterations as applicable. This can vary depending on the mode because autoseg modes would have no points!

        Image directory which contains all of the images as nii.gz files, and a folder containing the predicted labels and the ground truth label (denoted original)
        Inference task: The name of the inference run for which the scores are being computed, i.e. editing or initialisation only.
        Results save dir: The base directory which the results will all be saved to.
        weightmap_parametrisations: The dict containing the weightmap type names, and the corresponding parametrisations being used across all of the points in this computation.
        human_measure: The measure of performance, i.e. local responsiveness or temporal consistency.
        base_metric: The base metric being used for performance evaluation.
        per_class_scores: The bool which tells us whether we are only generating multi-class scores, or whether we also output a dict of class separated scores ALSO.
        config_labels: The dict of class label - class code relationships.
        sequentiality mode: The mode which the click sets are assumed to be in (E.g. CIM/accumulated across iters, or SIM/only the clicks for the current iter.)
        '''
        assert type(scoring_tools) == dict 
        assert type(infer_run_mode) == list, "Infer run config was not provided as a list"
        assert len(infer_run_mode) == 3, "Infer run config was not of the appropriate setup"
        assert type(weightmap_parametrisations) == dict 
        assert type(human_measure) == str 
        assert type(base_metric) == str
        assert type(per_class_scores) == bool 
        assert type(include_background_metric) == bool 
        assert type(config_labels) == dict 
    
        
        assert sequentiality_mode in ['CIM', 'SIM'], "The sequentiality mode was not supported for metric computation."

        assert human_measure.title() == "Temporal Non Worsening", "Human measure did not match the computation function"

        #In this case, the prediction, the ground truth, and the guidance points set, and the parametrisations for the weightmap for the metric are provided (only for a given iter). 

        if sequentiality_mode == "SIM":
            
            self.sim_temporal_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)
        
        if sequentiality_mode == "CIM":

            self.cim_temporal_computation(scoring_tools, infer_run_mode, img_directory_path, results_save_dir, weightmap_parametrisations, human_measure, base_metric, per_class_scores, include_background_metric, config_labels, sequentiality_mode)

                                           




    def __call__(self):
    
        inference_config_dict = dict() 

        inference_config_dict['app_dir'] = self.app_dir_path 
        inference_config_dict['inference_run_config'] = self.infer_run_mode
        
        inference_config_dict['dataset_name'] = self.studies
        inference_config_dict['dataset_subset'] = self.dataset_subset + f'_{self.infer_simulation_type}'
        
        inference_config_dict['datetime'] = self.datetime
        inference_config_dict['checkpoint'] = self.checkpoint

        inference_config_dict['inference_click_parametrisation'] = self.infer_run_parametrisation 
        inference_config_dict['run_infer_string'] = f'run_{self.infer_run_num}'
        
        metric_config_dict = dict() 
        metric_config_dict['click_weightmap_types'] = list(self.click_weightmaps_dict.keys())
        
        #The click parametrisations dict is re-assigned, which we will later save alongside the computed metrics in their corresponding folders for reference later.    
        metric_config_dict['click_weightmap_parametrisations'] = self.click_weightmaps_dict 
        

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
        
        if self.sequentiality_mode not in supported_sequentiality_modes:
            raise ValueError("The selected sequentiality mode (e.g. CIM) was not supported")
        
        if self.infer_simulation_type not in supported_simulation_types:

            raise ValueError("The selected simulation type (e.g. probabilistic) was not supported")
        
        metric_config_dict['gt_weightmap_types'] = self.gt_weightmap_types
        metric_config_dict['human_measure'] = self.human_measure 
        metric_config_dict['base_metric'] = self.base_metric 

        ################################################################################################################################################################################

        #Generation of the paths required for extracting the (segmentations, guidance point sets, guidance point parametrisations etc.) and the path for saving the results.

        path_generation_class = path_generation(inference_config_dict, metric_config_dict)

        inference_output_dir_path, results_save_dir = path_generation_class()

        ########################################################################################################################################


        '''
        Initialisation of the score computation tool:
        ''' 

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

        #############################################################################################

        #Initialising the score generation tool. 

        # weightmap_types = list(self.click_weightmaps_dict.keys()) + self.gt_weightmap_types 
        click_weightmap_types = list(self.click_weightmaps_dict.keys())
        gt_weightmap_types = self.gt_weightmap_types

        if self.base_metric == 'Dice':
            #IF there is an autoseg mode, OR, if we are computing for an autoseg init. then we need to allow for side-stepping of the fact that there is no guidance points/parametrisations applicable

            if self.infer_run_mode[0].title() == "Autoseg":
                metric_computer_tools = {"Autoseg":score_tool(config_labels, "None", ["None"], ["None"], self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores)}
            
            if len(self.infer_run_mode) > 1 and self.infer_run_mode[1].title() == "Autoseg":
                metric_computer_tools = {"Autoseg":score_tool(config_labels, "None", ["None"], ["None"], self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores),
                                        "Editing":score_tool(config_labels, self.human_measure, click_weightmap_types, gt_weightmap_types, self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores)
                                        }   
            
            if len(self.infer_run_mode) > 1 and self.infer_run_mode[1].title() == "Interactive":
                metric_computer_tools = {"Interactive":score_tool(config_labels, self.human_measure, click_weightmap_types, gt_weightmap_types, self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores),
                                        "Editing":score_tool(config_labels, self.human_measure, click_weightmap_types, gt_weightmap_types, self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores)
                                        } 
            if self.infer_run_mode[0].title() == "Interactive":

                metric_computer_tools = {"Interactive":score_tool(config_labels, self.human_measure, click_weightmap_types, gt_weightmap_types, self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores)}

        elif self.base_metric == 'Error Rate':
            #Currently this is only supported/intended for cross-iteration temporal non-worsening, therefore support is only provided for that infer mode.
            assert len(self.infer_run_mode) >  1

        
            metric_computer_tools = {
                                    "Editing":score_tool(config_labels, self.human_measure, click_weightmap_types, gt_weightmap_types, self.base_metric, self.include_background_mask, self.include_background_metric, self.ignore_empty, self.per_class_scores)
                                    } 

        #####################################################################################################################################################

        #Create the different permutations depending on the human measure! 

        if self.human_measure == "None":
            
            self.base_score_computation(metric_computer_tools,
                                        self.infer_run_mode,
                                        inference_output_dir_path,
                                        results_save_dir,
                                        self.click_weightmaps_dict,
                                        self.human_measure,
                                        self.base_metric, 
                                        self.per_class_scores,
                                        self.include_background_metric,
                                        config_labels,
                                        self.sequentiality_mode)

        elif self.human_measure == "Local Responsiveness":

            self.local_score_computation(metric_computer_tools,
                                        self.infer_run_mode,
                                        inference_output_dir_path,
                                        results_save_dir,
                                        self.click_weightmaps_dict,
                                        self.human_measure,
                                        self.base_metric,
                                        self.per_class_scores,
                                        self.include_background_metric,
                                        config_labels,
                                        self.sequentiality_mode
                                        )

        elif self.human_measure == "Temporal Non Worsening":

            self.temporal_consist_score_computation(metric_computer_tools,
                                                    self.infer_run_mode,
                                                    inference_output_dir_path,
                                                    results_save_dir,
                                                    self.click_weightmaps_dict,
                                                    self.human_measure,
                                                    self.base_metric,
                                                    self.per_class_scores,
                                                    self.include_background_metric,
                                                    config_labels,
                                                    self.sequentiality_mode
                                                    )
