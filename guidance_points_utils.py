import copy 
import os 
import json 

def guidance_dict_info(guidance_json_folder, image_name_no_ext, iteration_info, weightmap_parametrisations, sequentiality_mode, class_label_configs):
        
        '''
        Input: 
        Guidance Json Folder: Folder path which contains the json files for the guidance points at each iteration.
        
        Image name without the extension (Str) 

        Iteration_Infos: The nested list containing info about the iterations under consideration
        e.g. is it an editing iteration or an initialisation iteration, and the name of the iteration (e.g. iteration_1, final_iteration, interactive [initialisation], autoseg)
        
        Weightmap paramterisation: The dict containing the weightmaps being used, and their corresponding parametrisations which is to be used for every single point.
        
        Sequentiality mode: The str containing whether the clicks are accumulated across iterations (CIM) or are only from the current set of clicks (SIM).
        
        class label configs: The dict containing the class-label and the class-codes.
        
        Returns: The class separated dict for the given image's corrective clicks.
        '''
        
        assert type(guidance_json_folder) == str 
        assert type(image_name_no_ext) == str 
        assert type(iteration_info) == list 
        assert type(weightmap_parametrisations) == dict 
        assert type(sequentiality_mode) == str 
        assert type(class_label_configs) == dict 
        assert len(iteration_info) <= 2
        
        iteration_types = [info[0] for info in iteration_info]
        iteration_names = [info[1] for info in iteration_info]
        #Iteration name should be the 

        edit_types = ['deepedit']
        init_types = ['autoseg', 'interactive']
        final_types = ['final']

        assert all(iteration_types) in edit_types + init_types + final_types, "The iteration type was not a valid one for the json file name"

        #Generating the guidance json paths and extracting the dicts containing the corrective clicks! 

        if sequentiality_mode == "SIM": 
            
            #In this case, the iteration types should be of length 1 (only the current iteration under consideration)
            if iteration_types[0] in edit_types:
                
                guidance_json_path = os.path.join(guidance_json_folder, f'{iteration_types[0]}_iteration_{iteration_names[0]}.json')   

            elif iteration_types[0] in init_types or iteration_types[0] in final_types:
                
                guidance_json_path = os.path.join(guidance_json_folder, f'{iteration_types[0]}.json')
                
            else:
                raise ValueError("Missing iteration type")
        
            #Extracting the guidance points dicts, in the case where it does not exist (which may occur for autoseg iterations) then we just generate a dict with empty click sets for each class.

            if os.path.exists(guidance_json_path):

                with open(guidance_json_path, 'r') as f:
                    all_image_guidances = copy.deepcopy(json.load(f))
                
                guidance_points_output_dict = copy.deepcopy(all_image_guidances[image_name_no_ext]) 

            else:
                guidance_points_output_dict = dict()

                for class_label in class_label_configs:
                    guidance_points_output_dict[class_label] = []


        
        elif sequentiality_mode == "CIM":
            #In this case, we require the prior and current iteration's names to compute the difference (in order to extract the corrective points)

            #There are different permissible permutations for this: Just the init (auto or interactive), the editing iters (and their corresponding number), the editing iters and "final"
            # AND even the init & "final" (when editing iters num = 1).

            if len(iteration_types) == 1:
                #In this case it is just the init.
                guidance_json_path_1 = ''
                guidance_json_path_2 = os.path.join(guidance_json_folder, f'{iteration_types[0]}.json')

            if all(iteration_types) in edit_types:
                #In this case then it is being compared between editing iterations
                guidance_json_path_1 = os.path.join(guidance_json_folder, f'{iteration_types[0]}_iteration_{iteration_names[0]}.json')   
                guidance_json_path_2 = os.path.join(guidance_json_folder, f'{iteration_types[1]}_iteration_{iteration_names[1]}.json')

            elif iteration_types[0] in init_types and iteration_types[1] in edit_types:
                #In this case then it is being compared between init and editing iteration (which is not final)

                guidance_json_path_1 = os.path.join(guidance_json_folder, f'{iteration_types[0]}.json')
                guidance_json_path_2 = os.path.join(guidance_json_folder, f'{iteration_types[1]}_iteration_{iteration_names[1]}.json')
                
            elif iteration_types[0] in init_types and iteration_types[1] in final_types:
                #In this case it is between an init type and a final type (i.e. where there is one iteration of editing)
                guidance_json_path_1 = os.path.join(guidance_json_folder, f'{iteration_types[0]}.json')
                guidance_json_path_2 = os.path.join(guidance_json_folder, f'{iteration_types[1]}.json')

            elif iteration_types[0] in edit_types and iteration_types[1] in final_types:
                #In this case it is between an editing iteration, and the final editing iteration.
                guidance_json_path_1 = os.path.join(guidance_json_folder, f'{iteration_types[0]}_iteration_{iteration_names[0]}.json')
                guidance_json_path_2 = os.path.join(guidance_json_folder, f'{iteration_types[1]}.json')

            else:
                raise ValueError("Missing iteration type")
        
            #Extracting the guidance points dicts, in the case where it does not exist (which may occur for autoseg iterations) then we just generate a dict with empty click sets for each class.

            if os.path.exists(guidance_json_path_1) and os.path.exists(guidance_json_path_2):

                with open(guidance_json_path_1, 'r') as f:
                    guidance_points_1 = copy.deepcopy(copy.deepcopy(json.load(f))[image_name_no_ext])
                
                with open(guidance_json_path_2, 'r') as f:
                    guidance_points_2 = copy.deepcopy(copy.deepcopy(json.load(f))[image_name_no_ext])
                

            elif os.path.exists(guidance_json_path_2) and not os.path.exists(guidance_json_path_1):
                #I.e. if init, then path 1 is non existent. In this case it requires path 2 (the init) to exist however, and so it can only apply to the interactive init.
                with open(guidance_json_path_2, 'r') as f:
                    guidance_points_2 = copy.deepcopy(json.load(f))

                guidance_points_1 = dict()

                for class_label in class_label_configs:
                    guidance_points_1[class_label] = []
                
            elif not os.path.exists(guidance_json_path_2) and not os.path.exists(guidance_json_path_1):
                #I.e. if init, then path 1 is non existent. If path 2 is also non existent it is possibly due to it being an autoseg init.
                
                guidance_points_1 = dict()

                for class_label in class_label_configs:
                    guidance_points_1[class_label] = []

                guidance_points_2 = dict()

                for class_label in class_label_configs:
                    guidance_points_2[class_label] = []


            #Finding the difference between iters:

            guidance_points_output_dict = dict()

            for class_label in class_label_configs.keys():
                
                #extracting the list of points for the given class across both dicts.
                guidance_points_output_dict[class_label] = [point for point in guidance_points_2[class_label] if point not in guidance_points_1[class_label]]


        
        #Generating the guidance points parametrisations in the dict, dict, nested list format (weightmap_type, class, nested list of points) 

        guidance_points_parametrisations_dict = guidance_parametrisation_generator(guidance_points_output_dict, weightmap_parametrisations, 'Uniform')

        #Here we will extract the guidance point dicts and generate the guidance point parametrisations for generating the weightmaps during score generation. 

        return guidance_points_output_dict, guidance_points_parametrisations_dict
    
def  guidance_parametrisation_generator(guidance_points_dict, weightmap_parametrisations, individual_or_uniform):

    '''
    Function which takes the dictionary of guidance points, the weightmap types (both click-based and gt-based) + parametrisations for a mask generator, and a bool which states whether the 
    parametrisation is to be used uniformly across points or provided individually for each point (the latter is HIGHLY unlikely)
    
    Returns:

    The guidance points parametrisations dict which is structured as: mask_type, class, nested list of points (parametrisations) with parametrisations in a list
    '''

    assert type(guidance_points_dict) == dict 
    
    for class_label in guidance_points_dict.keys():
        assert type(guidance_points_dict[class_label]) == list #Make sure each class has a nested list of points. 
    
    assert type(weightmap_parametrisations) == dict #This should be a dict with weightmaps, and their corresponding class separated dict of parametrisations for the nested list of points.

    supported_generators = ['Uniform']

    assert individual_or_uniform in supported_generators, "The generation of the guidance point parametrisation dict failed since it was not supported"

    guidance_params_output_dict = dict() 

    if individual_or_uniform.title() == "Uniform":
        
        for weightmap_type, parametrisations in weightmap_parametrisations.items():
            
            assert type(parametrisations) == list 

            weightmap_type_dict = dict() 

            for class_label, nested_list_of_points in guidance_points_dict.items(): 
                
                #Initialise the nested list for the current class label: 
                class_wide_params_list = [parametrisations] * len(nested_list_of_points) #Multiplied by the length of the nested list of points for each class.

                weightmap_type_dict[class_label] = class_wide_params_list 

            guidance_params_output_dict[weightmap_type] = weightmap_type_dict 
    
    return guidance_params_output_dict

            
