import os 
from pathlib import Path 

class path_generation():
    '''
    Class containing the methods which are used for the generation of paths during score generation.

    This includes the paths for extracting the folder/diecty containing segmentations, guidance points, guidance point parametrisations and also a path for saving the results according to the metrics/parametrisations used.
    
    '''

    def __init__(self, inference_conf_dict, metric_conf_dict):
        self.metric_conf_dict = metric_conf_dict 
        self.inference_conf_dict = inference_conf_dict 

    def extract_inference_paths(self, conf_dict):
        '''
        This function takes the inference config dictionary which includes the config code which determines what type of click parametrisation was implemented during inference, 
        and generates the paths to find the segmentation masks.
        '''

        #Asserting the supported inference click parametrisations:

        supported_inference_click_parametrisations = ['None',
                                                'Fixed Click Size',
                                                'Dynamic Click Size']
        
        assert type(conf_dict) == dict

        #The assumed format for the inference click parametrisation dict is {'type':'parametrisation', 'click_collection_mode':'mode'}
        
        assert type(conf_dict['inference_click_parametrisation']) == dict, 'Could not generate the inference folder paths since the inference click parametrisation information was not in dict format.'

        click_parametrisation_type = None 

        for click_param in supported_inference_click_parametrisations:
            if click_param in list(conf_dict['inference_click_parametrisation'].keys()):
                click_parametrisation_type = click_param 

        #This code extracts the actual click parametrisation (i.e. default, a fixed but selected click size, or a dynamic click size).
        # 

        assert click_parametrisation_type != None, "The parametrisation for the click size was not provided"



        
        if conf_dict['inference_run_config'][0].title() == "Editing":
            #In this case, we are implementing init. + editing inference runs.

            initialisation = conf_dict['inference_run_config'][1].title()
            num_editing_iters = conf_dict['inference_run_config'][2] 

            
            inference_run_subtype = conf_dict['dataset_name'] + f"/{conf_dict['dataset_subset']}/{conf_dict['datetime']}/{conf_dict['checkpoint']}/{initialisation}_initialisation_{num_editing_iters}_edit_iters"
            
            if click_parametrisation_type.title() == "None":

                #In this case, it is the default behaviour, there is no click size parametrisation.

                final_inference_run_subtype = os.path.join(inference_run_subtype, "No Click Param", conf_dict['run_infer_string'])


                

            if click_parametrisation_type.title() == "Fixed Click Size":
                #In this case, there is an additional part of the subpath which denotes FIXED click size, and the parametrisation across the image dimensions.

                final_inference_run_subtype = os.path.join(inference_run_subtype, conf_dict['inference_click_parametrisation'][click_parametrisation_type], conf_dict['run_infer_string'])

            elif click_parametrisation_type.title() == "Dynamic Click Size":
                #In this case, there is an additional part of the subpath which denotes DYNAMIC click size, the parametrisation will be saved into a dict to match the guidance points dict that is 
                # being saved. 
                
                final_inference_run_subtype = os.path.join(inference_run_subtype, conf_dict['inference_click_parametrisation'][click_parametrisation_type], conf_dict['run_infer_string'])
            

            inference_output_subdirectory = 'datasets/' + final_inference_run_subtype
        
        else:
            #In this case, we are implementing just an initialisation inference run.

            inference_run_subtype = conf_dict['dataset_name'] + f"/{conf_dict['dataset_subset']}/{conf_dict['datetime']}/{conf_dict['checkpoint']}/{conf_dict['inference_run_config'][0].title()}_initialisation"

            if click_parametrisation_type.title() == "None":
                
                final_inference_run_subtype = os.path.join(inference_run_subtype, "No Click Param", conf_dict['run_infer_string'])

            elif click_parametrisation_type.title() == "Fixed Click Size":
                #In this case, there is an additional part of the subpath which denotes FIXED click size, and the parametrisation across the image dimensions.

                final_inference_run_subtype = os.path.join(inference_run_subtype, conf_dict['inference_click_parametrisation'][click_parametrisation_type], conf_dict['run_infer_string'])

            elif click_parametrisation_type.title() == "Dynamic Click Size":
                #In this case, there is an additional part of the subpath which denotes DYNAMIC click size, the parametrisation will be saved into a dict to match the guidance points dict that is 
                # being saved.
                final_inference_run_subtype = os.path.join(inference_run_subtype, conf_dict['inference_click_parametrisation'][click_parametrisation_type], conf_dict['run_infer_string'])

            inference_output_subdirectory = os.path.join('datasets', final_inference_run_subtype) 
        
        full_path = os.path.join(conf_dict['app_dir'], inference_output_subdirectory)

        # assert os.path.exists(full_path), 'The subdirectory does not exist!'

        return full_path, final_inference_run_subtype 
    

    def generate_results_paths(self, conf_dict_1, conf_dict_2, inference_run_subtype):
        '''
        This function takes the inference run config dict (conf_dict_1), metric config dictionary (conf_dict_2), the inference run subtype (which contains the dataset, inference run,
        run number etc) and generates the path for saving the results.

        '''
        #Generating the paths for saving the results. 

        '''
        Currently supported weightmap types and their corresponding parametrisations:

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
        
        When NONE, then we will just put NONE for both the weightmap and the parametrisation. 
        For weightmaps which do not have any parametrisation, then NONE will be the value also.
        '''
        click_weightmap_type_path =  'click_weightmaps' + "".join([f"_{subtype}_{conf_dict_2['click_weightmap_parametrisations']}" for subtype in conf_dict_2['click_weightmap_types']])
        gt_weightmap_type_path = 'gt_weightmaps' + "".join([f"_{subtype}" for subtype in conf_dict_2['gt_weightmap_types']])


        #Putting all of the configurations together into a path.
        full_metric_configuration_path = os.path.join(conf_dict_2['base_metric'], f"human_measure_{conf_dict_2['human_measure']}", click_weightmap_type_path, gt_weightmap_type_path)

        #obtaining the full path for saving the results

        #Extracting just the datetime/inference run info, not the dataset name etc again 
        
        infer_run_subtype_info = str(Path(*Path(inference_run_subtype).parts[2:]))
        results_save_dir = os.path.join(conf_dict_1['app_dir'], 'datasets', conf_dict_1["dataset_name"], f"{conf_dict_1['dataset_subset']}_results", full_metric_configuration_path, infer_run_subtype_info) 

        # if os.path.exists(results_save_dir):
        #     os.rmdir(results_save_dir)

        # os.makedirs(results_save_dir) 

        return results_save_dir


    def generate_guidance_points_path(self, inference_output_subdirectory):
        return os.path.join(inference_output_subdirectory, 'labels', 'guidance_points')
    
    def generate_guidance_point_parametrisations_path(self, inference_output_subdirectory):
        return os.path.join(inference_output_subdirectory,'labels','guidance_points_parametrisations')



    def __call__(self): 
        '''
        Inputs: 
        
        Metric configuration dict which contains all of the information about the selected metric, the selected weightmaps, and the parametrisations used for the generation of the image masks
        (if applicable for the click-based weightmap).
        
        Inference configuration dict which contains all of the information about the save folder (the dataset selected, the checkpoint, the inference run type [editing, auto init, interactive init], 
        the inference run (number) for probabilistic inference, and click parametrisation configuration during inference, the app directory, the test set used for inference (validation/hold-out)


        Returns: The dataset name, the subpath within the "datasets" folder in the app directory which gives the directory of segmentation masks used for score computation, and the subpath 
        from the app directory for the saving of the results.
        '''
        assert type(self.inference_conf_dict) == dict 
        assert type(self.metric_conf_dict) == dict 

        conf_dict_1 = self.inference_conf_dict
        conf_dict_2 = self.metric_conf_dict 

        segmentations_directory, inference_run_subtype = self.extract_inference_paths(conf_dict_1)

        results_save_directory = self.generate_results_paths(conf_dict_1, conf_dict_2, inference_run_subtype)


        return segmentations_directory, results_save_directory  

