import os

import argparse 
from os.path import dirname as up
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
import csv 
import json 
import sys
import shutil 
import copy

file_dir = os.path.join(os.path.expanduser('~'), 'MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus')
sys.path.append(file_dir)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class dice_score_tool():
    def __init__(self, label_names, original_dataset_labels, label_mapping):
    
        from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose
            )
        from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd  
        from monai.metrics import DiceHelper
        from monai.metrics import DiceMetric
        from monai.utils import MetricReduction

        self.original_dataset_labels = original_dataset_labels
        self.label_names = label_names
        self.label_mapping = label_mapping

        self.transforms_list = [
        LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=("pred", "gt")),
        Orientationd(keys=("pred", "gt"), axcodes="RAS"),  
        MappingLabelsInDatasetd(keys="gt", original_label_names=self.original_dataset_labels, label_names = self.label_names, label_mapping=self.label_mapping)
        ]

        self.transforms_composition = Compose(self.transforms_list, map_items = False)

        self.dice_computation_class = DiceHelper(  # type: ignore
                include_background= False,
                sigmoid = False,
                softmax = False, 
                activate = False,
                get_not_nans = False,
                reduction = MetricReduction.NONE, #MetricReduction.MEAN,
                ignore_empty = True,
                num_classes = None
        )
        #self.dice_computation_class = DiceMetric()

    def __call__(self, pred_folder_path, gt_folder_path, image_name):

        pred_image_path = os.path.join(pred_folder_path, image_name)
        gt_image_path = os.path.join(gt_folder_path, image_name)

        input_dict = {"pred":pred_image_path, "gt":gt_image_path}
        output_dict = self.transforms_composition(input_dict)


        dice_score = self.dice_computation_class(y_pred=output_dict["pred"].unsqueeze(0), y=output_dict["gt"].unsqueeze(0))
        #print(dice_score[0])
        return float(dice_score[0]) #float(dice_score) #float(dice_score[0])

def dice_score_computation(img_directory, inference_tasks, results_save_dir, jobs, study_name):
    '''
    This method should compute the dice scores for the set of images and task which has been provided. It should 
    #save this in a csv file, the dice scores returned should provide the dice scores across all the sets of images
    #that have been provided. 
    '''
    if "compute" in jobs:
        import re 

        #Implement the computation scripts here.

        #Here we will generate the paths for the images with which we want to compute dice scores
        
        #Obtaining list of image names, not done in numeric order:
        image_names = [x for x in os.listdir(img_directory) if x.endswith('.nii.gz')]
        gt_image_folder = os.path.join(img_directory, 'labels', 'original')
        final_image_folder = os.path.join(img_directory,'labels', 'final')
        framework = inference_tasks[0]
        if framework == "deepeditplusplus":
            framework = "deepeditPlusPlus"

        dataset_name = study_name[:-9]

        label_config_path = os.path.join(file_dir, "monailabel", framework, dataset_name + '_label_configs.txt')
        
        ################### Importing the label configs dictionary #####################

        with open(label_config_path) as f:
            config_dict = json.load(f)

        config_labels = config_dict["labels"]
        config_original_dataset_labels = config_dict["original_dataset_labels"]
        config_label_mapping = config_dict["label_mapping"]

        if os.path.exists(results_save_dir) == True:
            shutil.rmtree(results_save_dir)
        os.makedirs(results_save_dir)

        dice_computer = dice_score_tool(config_labels, config_original_dataset_labels, config_label_mapping)

        if inference_tasks[1] == "deepedit":
            
            initialisation_folder = os.path.join(img_directory, 'labels', inference_tasks[2])
            

            iteration_folders = [x for x in os.listdir(os.path.join(img_directory, 'labels')) if x.startswith('deepedit_iteration')]
            iteration_folders.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
            #sorts the folders by true numeric iteration value, not by the standard method in python when using strings.
            #this is needed because the iterations need to be in order.

            for image in image_names:
                dice_scores = [image]
                
                 
                dice_scores.append(dice_computer(initialisation_folder, gt_image_folder, image)) # Adding the initialisation dice score
                
                for iteration_folder in iteration_folders: #Adding the dice scores on the intermediary iterations
                    dice_scores.append(dice_computer(os.path.join(img_directory, 'labels', iteration_folder), gt_image_folder, image))
                
                #for the final image
                dice_scores.append(dice_computer(final_image_folder, gt_image_folder, image))
                #if np.any(np.array(dice_scores[1:]) < 0.5):
                    #print(dice_scores)
                    #print(image)
                with open(os.path.join(results_save_dir, 'dice_score_results.csv'),'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(dice_scores)

        else:
            gt_image_folder = os.path.join(img_directory, 'labels', 'original')

            for image in image_names:
                dice_scores = [image]
                dice_scores.append(dice_computer(final_image_folder, gt_image_folder, image))

                with open(os.path.join(results_save_dir, 'dice_score_results.csv'),'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(dice_scores)
            # return dice_scores 

def dice_score_extraction(results_save_dir):
    with open(os.path.join(results_save_dir, 'dice_score_results.csv'), newline='') as f:
        dice_score_reader = csv.reader(f, delimiter=' ', quotechar='|')
        first_row = f.readline()
        first_row = first_row.strip()
        #print(first_row)
        #n_cols = first_row.count(',') + 1 
        dice_scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
        #print(dice_scores)
        for row in dice_score_reader:
            row_str_list = row[0].split(',')
            #print(row_str_list)
            for index, string in enumerate(row_str_list):
                if index > 0:
                    dice_scores[index].append(float(string))
                elif index == 0:
                    dice_scores[index].append(string)
    return dice_scores 

def dice_score_collection(results_save_dir, image_subtasks, dice_score_files_base_dir, rejection_value=0):
    #obtaining the paths for all of the dice score files we want to merge together:
    dice_score_paths = [os.path.join(dice_score_files_base_dir, image_subtask) for image_subtask in image_subtasks]

    #extracting the dice scores and collecting them together.
    all_dice_scores = []

    for dice_path in dice_score_paths:
        with open(os.path.join(dice_path, 'dice_score_results.csv'), newline='') as f:
            dice_score_reader = csv.reader(f, delimiter=' ', quotechar='|')
            first_row = f.readline()
            first_row = first_row.strip()
            #print(first_row)
            #n_cols = first_row.count(',') + 1 
            dice_scores = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
            #print(dice_scores)
            for row in dice_score_reader:
                row_str_list = row[0].split(',')
                #print(row_str_list)
                for index, string in enumerate(row_str_list):
                    if index > 0:
                        dice_scores[index].append(float(string))
                    elif index == 0:
                        dice_scores[index].append(string)

            all_dice_scores.append(dice_scores)
        
    final_output_dice_scores = all_dice_scores[0]

    for dice_score_set in all_dice_scores[1:]:
        #dice_score_set_index = index + 1
        #current_dice_score_set = all_dice_scores[dice_score_set_index]

        for output_index in range(len(final_output_dice_scores)):
            
            final_output_dice_scores[output_index] += dice_score_set[output_index] #current_dice_score_set[output_score_index]


    #Here we will compute the median of all the columns so that we can view them when necessary, without needing to keep running the plotting script.
    #Also use can use this median as part of the plotting function. 

    # dice_score_average = ['averages']
    non_rejected_rows = []
    for dice_score_row_index in range(len(final_output_dice_scores[0])): #final_output_dice_scores[1:]:

        dice_score_row = [final_output_dice_scores[j][dice_score_row_index] for j in range(len(final_output_dice_scores))]
        if all(dice_score_row[1:]) >= rejection_value:
        
        # accepted_dice_scores = [i for i in dice_score_column if i >= rejection_value]
            non_rejected_rows.append(dice_score_row[1:])

        # dice_score_average.append(sum(accepted_dice_scores)/len(accepted_dice_scores))
    # totals = copy.deepcopy(non_rejected_rows[0])
    # count = 1
    # for non_rejected_row in non_rejected_rows[1:]:
    #     for index, val in enumerate(non_rejected_row):
    #         totals[index] += val 
    #     count += 1
    
    # for total in totals:
    #     dice_score_average.append(total/count)
    

    # for i in range(len(dice_score_average)):
    #     final_output_dice_scores[i].append(dice_score_average[i])

    #compute standard deviations. 
    # stdevs = np.std(np.array(non_rejected_rows), axis=0)
    #appending them to the file

    # final_output_dice_scores[0].append('stdevs')
    # for i in range(len(stdevs)):
    #     final_output_dice_scores[i + 1].append(stdevs[i])

    if os.path.exists(results_save_dir) == True:
        shutil.rmtree(results_save_dir)
    os.makedirs(results_save_dir, exist_ok=True)
    
    with open(os.path.join(results_save_dir, 'dice_score_results.csv'),'a') as f:
        writer = csv.writer(f)
        
        for i in range(len(final_output_dice_scores[0])):
            output_row = [sublist[i] for sublist in final_output_dice_scores]
            writer.writerow(output_row)

    return 

def dice_visualisation(dice_scores, task_configs, results_dir, run_infer_string, rejection_value):
    import seaborn as sb
    import pandas as pd
    inference_task = task_configs[1]
    print(f'Inference task is {inference_task}')
    if inference_task == "deepedit":
        initialisation = task_configs[2]
        num_clicking_iters = task_configs[3] 
        task_title = f'{initialisation.capitalize()} initialisation, {num_clicking_iters} {inference_task.capitalize()} iterations'
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title(f'{task_title[:-1]} dice scores')
        ax.set_xlabel('Clicking Iteration')
        ax.set_ylabel('Dice Score')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        task_title = inference_task.capitalize()
        ax.set_title(f'{inference_task.capitalize()} Dice Scores')
        ax.set_ylabel('Dice Score')
    
    # If we are doing a deepedit output, then we just want a moving average across the iterations (possibly with some range bars?)

    # If we are doing an autoseg or deepgrow output, then we just want a distribution of the scores. 

    ################ Removing the failure cases from the mean computation, and printing the list of failure cases into a separate text file to use as exemplars. ###################
    #print(np.array(dice_scores).shape)
    tmp_array = np.array(dice_scores)
    # print(tmp_array[0, :])
    dice_scores_array = tmp_array[1:,:].astype(np.float64)
    dice_scores_array = dice_scores_array.T
    # print(dice_scores_array[-1, :])
    #We remove the average row because its not necessary.
    dice_scores_array = dice_scores_array[:, :]
    # print(dice_scores_array[-1, :])
    image_names = tmp_array[0,:]
    # print(image_names)

    #Setting failure case dice score:
    failure_dice = rejection_value
    # print(dice_scores_array.shape)
    failure_images = dict()
    row_removal_indices = []
    #We do it up till shape -2 since we do not want to include the stdev and mean in the row removal list.
    for index in range(dice_scores_array.shape[0] - 2): #np.array(dice_scores):
        sub_array = dice_scores_array[index, :]
        #print(sub_array)
        if np.any(sub_array < failure_dice):
            
            # print(sub_array)
            # print(image_names[index])

            failure_images[image_names[index]] = sub_array.tolist()
            row_removal_indices.append(index)
    
    #dice_scores = [i if i >= failure_dice in dice_scores]
    #dice_scores = []
    ################################################################################################################################################################################

    ######################### For the initialisation failures/general failures, save the names of the images #######################################################

    with open(os.path.join(results_dir,'failure_cases.txt'), 'w') as f:
        f.write(json.dumps(failure_images))
    
    #Base directory which we will place the folder in for saving plots:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(results_dir))), 'Plots', "Standalone")
    folder_save_name = f"model_{datetime}_checkpoint_{checkpoint}/{run_infer_string}"


    save_dir = os.path.join(base_dir, folder_save_name)
    os.makedirs(save_dir, exist_ok=True)

    ###### Remove the failure case rows ##########
    # print(np.sum(dice_scores_array, axis=0)/249)
    # print(dice_scores_array[row_removal_indices,:][0][0])

    final_dice_scores_array = np.delete(dice_scores_array, row_removal_indices, axis=0)

    final_dice_scores_array_scores = final_dice_scores_array[:-2,:]
    # print(final_dice_scores_array[-3:, :])
    dice_scores_mean = final_dice_scores_array[-2, :]
    dice_scores_stdev = final_dice_scores_array[-1, :]

    # print(final_dice_scores_array.shape)
    ####################################################################################################################################

    if inference_task == "deepedit":
        x = np.array(range(final_dice_scores_array_scores.shape[1]))
        y_averages = dice_scores_mean #np.mean(final_dice_scores_array, axis= 0)
        print(y_averages)
        # print(np.sum(final_dice_scores_array, axis=0))
        y_mins = np.min(final_dice_scores_array_scores, axis = 0)
        y_maxes = np.max(final_dice_scores_array_scores, axis = 0) 
    
        y_errors_min = abs(y_averages - y_mins)
        y_errors_max = abs(y_averages - y_maxes)

        y_errors = np.concatenate((np.expand_dims(y_errors_min, axis=0), np.expand_dims(y_errors_max, axis=0)))
    

        ax.errorbar(x, y_averages, yerr = y_errors, capsize=5, fmt='ro--', ecolor='blue')#color=colours[colour_counter])
        ax.set_xticks(x)
        # colour_counter += 1
        # print(colour_counter)
        # colour_counter = 0
        # x = x[0]
        # for experiment_key in list(moving_averages.keys()):
            
        #     experiment_dict = moving_averages[experiment_key]
        #     update_keys = list(experiment_dict.keys())
        #     update_keys.sort(key=int)
        #     for update_key in update_keys:
        #         #print(list(experiment_dict.keys()))
        #         print(experiment_dict[update_key])
        #         ax.plot(x, experiment_dict[update_key], color = colours[colour_counter])
        #         colour_counter += 1
                
        
        #ax.legend(loc = 'below')
        # ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
        #             bbox_transform=fig.transFigure, ncol=4)
        ax.set_ylim(bottom=np.min(y_mins - 0.05), top=1.0)

        #ax.scatter(x, np.array(dice_scores))

        
        plt.savefig(os.path.join(save_dir, "deepedit_iterations.png"))
        plt.show()
    else:
        x = np.array(range(len(final_dice_scores_array_scores)))
        y = np.array(final_dice_scores_array_scores)

        ax.set_ylim(bottom=max(0, np.min(y) - 0.2), top=min(np.max(y) + 0.2, 1)) #1.0)

        print(x.shape)
        print(y.shape)

        dataset = pd.DataFrame({
            "value":np.squeeze(y)
        })
        # ax.scatter(x, y)
        # plt.show()

        sb.swarmplot(data=dataset["value"], size=5)
        sb.boxplot(y="value", data=dataset, whis=2.0)
        
        plt.savefig(os.path.join(save_dir, "initialisation.png"))
        plt.show()
def dice_comparison_visualisation(dice_scores_nested, task_configs, result_dirs, inference_image_subtasks, comparison_type, datetimes, checkpoints, plot_type, run_infer_string, rejection_value):
    # import seaborn as sb
    import pandas as pd
    inference_task = task_configs[1]
    #print(f'Inference task is {inference_task}')
    if inference_task == "deepedit":
        initialisation = task_configs[2]
        num_clicking_iters = task_configs[3] 
        task_title = f'{initialisation.capitalize()} initialisation, {num_clicking_iters} {inference_task.capitalize()} iterations'
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_title(f'{task_title[:-1]} dice scores')
        ax.set_xlabel('Clicking Iteration')
        ax.set_ylabel('Dice Score')
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        task_title = inference_task.capitalize()
        ax.set_title(f'{inference_task.capitalize()} Dice Scores')
        ax.set_ylabel('Dice Score')
    
    #Tracking the absolute y_min for the error bars
    tracking_y_min = [1] * (int(num_clicking_iters) + 1)
    #Tracking the averages across all to determine the y_average min.
    y_averages_tracking = np.array([1] * (int(num_clicking_iters) + 1))


    colour_counter = 0
    dot_format = ["rx--", "bx--", "gx--", "cx--", "kx--", "mx--"] 
    c_map = ["blue", "green", "red", "purple", "orange", "black"]

    ########### Loading in the model summaries for the models ##############

    with open("/home/parhomesmaeili/Model_Config_Details/model_summary_details.txt") as f:
        dictionary = f.read()
        #print(dictionary)
        summary_details = json.loads(dictionary)

    #n_comparisons = len(result_dirs)
    #print(n_comparisons)

    ## Tracking all the failures across the different comparison models to examine them side-by-side ###
    all_failures = dict()
    print(result_dirs)
    for index, result_dir in enumerate(result_dirs):
        # If we are doing a deepedit output, then we just want a moving average across the iterations (possibly with some range bars?)

        # If we are doing an autoseg or deepgrow output, then we just want a distribution of the scores. 

        ################ Removing the failure cases from the mean computation, and printing the list of failure cases into a separate text file to use as exemplars. ###################
        #print(np.array(dice_scores).shape)
        #print(result_dir)
        #print(inference_image_subtasks[index])

        dice_scores = dice_scores_nested[index]

        tmp_array = np.array(dice_scores)

        dice_scores_array = tmp_array[1:,:].astype(np.float64)
        dice_scores_array = dice_scores_array.T 

        #
        dice_scores_array = dice_scores_array[:, :]
        image_names = tmp_array[0,:]

        #Setting failure case dice score:
        failure_dice = rejection_value
        #print(dice_scores_array.shape)
        failure_images = dict()
        row_removal_indices = []
        #It is done up till shape - 2 because we do not want to count the average and stdev as samples.
        for dice_row in range(dice_scores_array.shape[0] - 2): #np.array(dice_scores):
            sub_array = dice_scores_array[dice_row, :]
            #print(sub_array)
            if np.any(sub_array < failure_dice):
                # print('failure_row')
                # print(sub_array)
                #print(image_names[index])

                failure_images[image_names[dice_row]] = sub_array.tolist()
                row_removal_indices.append(dice_row)

        #Adding the list of failures to the total comparison dict 
        #print(checkpoints)
        #print(datetimes)
        #print(index)
        if comparison_type == "checkpoint":
            all_failures[checkpoints[index]] = failure_images 
        elif comparison_type == "model":
            all_failures[datetimes[index]] = failure_images 
        # all_failures[inference_image_subtasks[index]] = failure_images

        #dice_scores = [i if i >= failure_dice in dice_scores]
        #dice_scores = []
        ################################################################################################################################################################################

        ######################### For the initialisation failures/general failures, save the names of the images #######################################################

        # with open(os.path.join(result_dir,'failure_cases_comparison_loop.txt'), 'w') as f:
        #     f.write(json.dumps(failure_images))
        

        ###### Remove the failure case rows ##########

        final_dice_scores_array = np.delete(dice_scores_array, row_removal_indices, axis=0)

        final_dice_scores_array_scores = final_dice_scores_array[:-2,:]
        dice_scores_mean = final_dice_scores_array[-2, :]
        dice_scores_stdev = final_dice_scores_array[-1, :]
        ####################################################################################################################################

        if inference_task == "deepedit":
            x = np.array(range(final_dice_scores_array_scores.shape[1]))
            y_averages = dice_scores_mean #np.mean(final_dice_scores_array_, axis= 0)
            print(y_averages)
            y_mins = np.min(final_dice_scores_array_scores, axis = 0)
            y_maxes = np.max(final_dice_scores_array_scores, axis = 0) 
        
            y_errors_min = abs(y_averages - y_mins)
            y_errors_max = abs(y_averages - y_maxes)

            y_errors = np.concatenate((np.expand_dims(y_errors_min, axis=0), np.expand_dims(y_errors_max, axis=0)))

            if plot_type == "errorbar":
                if comparison_type == "checkpoint":
                    ax.errorbar(x, y_averages, yerr = y_errors, capsize=8, elinewidth=2, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=checkpoints[index])
                elif comparison_type == "model":
                    ax.errorbar(x, y_averages, yerr = y_errors, capsize=8, elinewidth=2, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=summary_details[datetimes[index]])
                # ax.errorbar(x, y_averages, yerr = y_errors, capsize=5, fmt=dot_format[colour_counter], ecolor=c_map[colour_counter], label=) #ecolor='blue')#color=colours[colour_counter])
            else:
                if comparison_type == "checkpoint":
                    ax.plot(x, y_averages, dot_format[colour_counter], label=checkpoints[index])
                elif comparison_type == "model":
                    ax.plot(x, y_averages, dot_format[colour_counter], label=summary_details[datetimes[index]])
                   
            
            # print(c_map(colour_counter))
            colour_counter += 1
            for ind, val in enumerate(y_mins.tolist()): # y_mins < tracking_y_min:
                if tracking_y_min[ind] > val:
                    tracking_y_min[ind] = val 

            y_averages_tracking = np.concatenate((y_averages_tracking, y_averages)) 

        #ax.scatter(x, np.array(dice_scores))
    
    #Saving all the failure cases: 
    #For the comparisons we need to save them in a separate folder:
    
    #Base directory which we will place the folder in:
    # base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(result_dir))), 'Plots', task_title, "Comparison", comparison_type)

    #We have different base directories depending on what is being compared. If it is the checkpoints being compared, then it would be within the same model.
    #If it is models being compared, then it needs to be saved outside of the individual model folders.
    if comparison_type == "checkpoint":
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(result_dir))), 'Plots', "Comparison", comparison_type)
        folder_save_name = f"model_{datetimes[0]}/checkpoints"
        for checkpoint in checkpoints:
            folder_save_name += f"_{checkpoint}"
    elif comparison_type == "model":
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(result_dir)))), 'Plots', "Comparison", comparison_type)
        folder_save_name = f"checkpoint_{checkpoints[0]}/models"
        for model in datetimes:
            folder_save_name += f"_{model}"

    save_dir = os.path.join(base_dir, f'{folder_save_name}/{run_infer_string}')
    os.makedirs(save_dir, exist_ok=True)
    #Saving the failure cases across the comparison sets

    with open(os.path.join(save_dir,'failure_cases_comparison_loop.txt'), 'w') as f:
            f.write(json.dumps(all_failures))

    if inference_task == "deepedit":
        ax.set_xticks(x)
        # colour_counter += 1
        # print(colour_counter)
        # colour_counter = 0
        # x = x[0]
        # for experiment_key in list(moving_averages.keys()):
            
        #     experiment_dict = moving_averages[experiment_key]
        #     update_keys = list(experiment_dict.keys())
        #     update_keys.sort(key=int)
        #     for update_key in update_keys:
        #         #print(list(experiment_dict.keys()))
        #         print(experiment_dict[update_key])
        #         ax.plot(x, experiment_dict[update_key], color = colours[colour_counter])
        #         colour_counter += 1
                
        
        #ax.legend(loc = 'below')
        ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
                    bbox_transform=fig.transFigure, ncol=4)
        
        if plot_type == "errorbar":
            ax.set_ylim(bottom=np.min(np.array(tracking_y_min) - 0.05), top=1.0)
        elif plot_type == "scatter":
            ax.set_ylim(bottom=max(0,np.min(y_averages_tracking) - 0.05), top=min(1, np.max(y_averages_tracking) + 0.05))

    
    plt.savefig(os.path.join(save_dir, "comparison_plot.png"), bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/imagesTs")
    parser.add_argument("--datetime", nargs="+", default=["31052024_195641"])
    parser.add_argument("--checkpoint", nargs="+", default=["best_val_score_epoch"])
    parser.add_argument("--infer_run", nargs="+", default=['0', '1', '2'])
    parser.add_argument("-ta", "--task", nargs="+", default=["deepeditplusplus","deepedit", "autoseg", "3"], help="The framework selection + subtask/mode which we want to execute")
    parser.add_argument("--app_dir", default = "MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus")
    parser.add_argument("--job", default= "plot_multiple", help="argument that determines which job is required from the script (i.e. plotting or computing dice scores)")
    parser.add_argument("--plot_type", default="errorbar")
    parser.add_argument("--rejection_val", default='0.5', help='Parameter which controls what the failure value is for the dice scores to not include in averages')
    #parser.add_argument("--models")
    args = parser.parse_args()

    app_dir = os.path.join(up(up(up(os.path.abspath(__file__)))), args.app_dir)
    framework = args.task[0]
    inference_task = args.task[1]
    
    dataset_name = args.studies[:-9]
    dataset_subset = args.studies[-8:]

    job = args.job
    
    #This is for computing the dice scores for a single run of a single model/single checkpoint.
    if job == "compute":
        #This is for the single checkpoint, single model, single run inference output processing.
        if len(args.datetime) == 1 and len(args.checkpoint) == 1 and len(args.infer_run) == 1:
            datetime = args.datetime[0]
            checkpoint = args.checkpoint[0]

            run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3] 

                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/{run_infer_string}'
                inference_image_subdirectory = 'datasets/' + inference_image_subtask
            else:
                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/{run_infer_string}'
                inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + '/results/' + inference_image_subtask 
            #print(results_save_dir)
            # os.makedirs(results_save_dir, exist_ok=True)
            
        dice_score_computation(os.path.join(app_dir, inference_image_subdirectory), args.task, results_save_dir, job, args.studies)

    #This is for collecting the dice scores across the different runs of a single model/single checkpoint.
    elif job == "collecting":

        datetime = args.datetime[0]
        checkpoint = args.checkpoint[0]
        run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

        if inference_task == "deepedit":
            initialisation = args.task[2]
            num_clicking_iters = args.task[3] 

            inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_{infer_run}' for infer_run in args.infer_run]
            # inference_image_subdirectory = 'datasets/' + inference_image_subtask
            results_save_dir = app_dir + '/results/' + dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
        else:
            inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_{infer_run}' for infer_run in args.infer_run] 
            # inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + '/results/' + dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
        dice_score_files_base_dir = app_dir + '/results/'
    
        dice_score_collection(results_save_dir, inference_image_subtasks, dice_score_files_base_dir, float(args.rejection_val))#, args.infer_run) 


    #This is for plotting a single RUN single MODEL/single checkpoint OR concatenated RUN single model/checkpoint 
    elif job == "plot_single_model":
        #This is for the single checkpoint, single model, single run inference output processing.
        if len(args.datetime) == 1 and len(args.checkpoint) == 1:
            datetime = args.datetime[0]
            checkpoint = args.checkpoint[0]

            #Here we will convert the list of runs into a single string which outlines the folder which we our collated dice score results are stored in.
            run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3] 

                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
                # inference_image_subdirectory = 'datasets/' + inference_image_subtask
            else:
                inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_collection/{run_infer_string}'
                # inference_image_subdirectory = 'datasets/' + inference_image_subtask

            results_save_dir = app_dir + f'/results/' + inference_image_subtask 
            #print(results_save_dir)
            # os.makedirs(results_save_dir, exist_ok=True)

        dice_scores = dice_score_extraction(results_save_dir) # dice_score_computation(os.path.join(app_dir, inference_image_subdirectory), args.task, results_save_dir, job, args.studies)
        
        dice_visualisation(dice_scores, args.task, results_save_dir, run_infer_string, float(args.rejection_val))
    
    #This is for plotting a single RUN multiple model/checkpoint OR for the concatenated results across runs for multiple models/checkpoints
    elif job == "plot_multiple":
        results_save_base_dir = app_dir + '/results/'
        #Here we will convert the list of runs into a single string which outlines the folder which we our collated dice score results are stored in.
        run_infer_string = 'run' + "".join([f"_{run}" for run in args.infer_run])

        if len(args.datetime) > 1:
            # dataset_name = args.studies[:-9]
            # dataset_subset = args.studies[-9:]

            #In this case we are plotting different models entirely against one another
            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3]
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{args.checkpoint[0]}/run_collection/{run_infer_string}' for datetime in args.datetime]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]

            else:
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{args.checkpoint[0]}/run_collection/{run_infer_string}' for datetime in args.datetime]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks] 
            #Comparison subtype is the model setting
            comparison_subtype = "model"

        elif len(args.checkpoint) > 1:
            #In this case then we are plotting different checkpoints against one another
            # dataset_name = args.studies[:-9]
            # dataset_subset = args.studies[-9:]

            if inference_task == "deepedit":
                initialisation = args.task[2]
                num_clicking_iters = args.task[3]
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{args.datetime[0]}/{checkpoint}/run_collection/{run_infer_string}' for checkpoint in args.checkpoint]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]
            else:
                inference_image_subtasks = [dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{args.datetime[0]}/{checkpoint}/run_collection/{run_infer_string}' for checkpoint in args.checkpoint]

                # inference_image_subdirectories = ['datasets/' + inference_image_subtask for inference_image_subtask in inference_image_subtasks]
            
            ##### Comparison subtype is : checkpoint
            comparison_subtype = "checkpoint"
            
        
        #Create the list of save directories to look into!
        results_save_dirs = [results_save_base_dir + f"{inference_image_subtask}" for inference_image_subtask in inference_image_subtasks]

        #print(results_save_dirs)
        #print(inference_image_subdirectories)

        dice_scores = []
        #Saving a nested list of dice scores
        for index, results_save_dir in enumerate(results_save_dirs):
            # os.makedirs(results_save_dir, exist_ok=True)
            #print(inference_image_subdirectories[index])
            dice_scores.append(dice_score_extraction(results_save_dir)) #dice_score_computation(os.path.join(app_dir, inference_image_subdirectories[index]), args.task, results_save_dirs[index], job, args.studies))
        
        dice_comparison_visualisation(dice_scores, args.task, results_save_dirs, inference_image_subtasks, comparison_subtype, args.datetime, args.checkpoint, args.plot_type, run_infer_string, float(args.rejection_val))