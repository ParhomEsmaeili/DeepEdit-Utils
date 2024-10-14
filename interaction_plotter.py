import os
from pydoc import classname
from re import L
import numpy as np
import json
from pathlib import Path 
from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose,
            ToNumpyd
            )

import sys
import copy 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import shutil
from monai.visualize import plot_2d_or_3d_image
import torch 
import csv 
import pandas as pd

file_dir = os.path.join(os.path.expanduser('~'), 'MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus')
sys.path.append(file_dir)

from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd, NormalizeLabelsInDatasetd, FindDiscrepancyRegionsDeepEditd

import argparse 
from os.path import dirname as up





def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name
    for name, value in locals().items():
        if value is var:
            return name
        



def generate_discrepancies(loading_output_dict, dictionary_segmentation_keys):
    '''
    Function which computes the discrepancies between each (iteration) segmentation and the GT. 
    '''
    discrepancy_transforms_list = [
        ToNumpyd(keys=("previous_seg", "label")),
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="previous_seg", discrepancy="discrepancy"),
        ]


    composed_transforms = Compose(discrepancy_transforms_list, map_items = False)
    
    output_dictionary = dict()

    ############ Here we compute the discrepancies across all of the iterations provided (always excludes the final seg. discrepancy)
    for index, key in enumerate(dictionary_segmentation_keys):
        #index is used for when we save the discrepancies, so that we can list out discrepancy_iter_{i}. For each segmentation:

        # create a copy just in case it is necessary, possible because we need to convert to numpyd.
        copy_transform_output = copy.deepcopy(loading_output_dict)
        # for key:val pairs in the list of key:val pairs, remove those which we are not needing, only keep the one which we do (for the segmentation we want)
        # this means removing the other segmentations and their meta_dicts, we still need the labels and mapping etc.
        removal_segmentations = ['image'] + [seg for seg in dictionary_segmentation_keys if seg != key]

        #We generate the list of keys that we want to delete: 
        delete_keys = [dict_key for dict_key in copy_transform_output.keys() if dict_key in removal_segmentations or dict_key in [seg + '_meta_dict' for seg in removal_segmentations]]

        for delete_key in delete_keys:

            del copy_transform_output[delete_key]

        #Replace the key that we want to compute discrepancy for, with "previous seg"
            
        copy_transform_output["previous_seg"] = copy.deepcopy(copy_transform_output[key])
        copy_transform_output["previous_seg_meta_dict"] = copy.deepcopy(copy_transform_output[f'{key}_meta_dict'])
        #keep the segmentation key for the current iteration, and change the name
        del copy_transform_output[key]
        del copy_transform_output[f'{key}_meta_dict']

        #Changing the "GT" key to "label" (these changes just help prevent confusion between sets of transforms)
            
        copy_transform_output["label"] = copy.deepcopy(copy_transform_output["GT"])
        copy_transform_output["label_meta_dict"] = copy.deepcopy(copy_transform_output["GT_meta_dict"])
        #keep the GT key and change the name 
        del copy_transform_output["GT"]
        del copy_transform_output["GT_meta_dict"]


        discrepancy_output_dict = composed_transforms(copy_transform_output)

        #
        final_discrepancy_output_dict = dict()
        for key in loading_output_dict["label_names"].keys():
            final_discrepancy_output_dict[key] = discrepancy_output_dict['discrepancy'][key]
        
        #saving the discrepancies to the total set of them across all iterations
        output_dictionary[f'discrepancy_iter_{index + 1}'] = final_discrepancy_output_dict #discrepancy_output_dict['discrepancy']

        #We only need to plot the false_negative discrepancies, since plotting all of the classes will cover all discrepancies (and even the original click simulation method
        #is technically clickinnnnnng on a subsection of the false-negative discrepancy when it does "other area" on false-positive correction clicks)

    
    return output_dictionary

def loading_images(dictionary_segmentation_keys, label_config_path, transform_input_dict):

    '''
    Function which reorientates the segmentations to match the RAS convention of the guidance points for plotting them. Also maps the GT labels to the task specification.

    '''
    with open(label_config_path) as f:
            config_dict = json.load(f)

    transform_input_dict["labels"] = config_dict["labels"]
    transform_input_dict["original_dataset_labels"] = config_dict["original_dataset_labels"]
    transform_input_dict["label_mapping"] = config_dict["label_mapping"]

  

    
    composed_transform = [
    LoadImaged(keys=dictionary_segmentation_keys + ["GT"], reader="ITKReader", image_only=False),
    EnsureChannelFirstd(keys=dictionary_segmentation_keys + ["GT"]),
    MappingLabelsInDatasetd(keys="GT", original_label_names = transform_input_dict["original_dataset_labels"], label_names = transform_input_dict["labels"], label_mapping=transform_input_dict["label_mapping"]),
    NormalizeLabelsInDatasetd(keys="GT", label_names=transform_input_dict["labels"]), 
    #We must orientate to RAS so that the guidance points are in the correct coordinate system for the inference script.
    Orientationd(keys=dictionary_segmentation_keys + ["GT"], axcodes="RAS"),
    ]

    transform_output_dict = Compose(composed_transform, map_items=False)(transform_input_dict)

    



    return transform_output_dict

def plot_tensorboard(input_image, initialisation, guidances, discrepancies_dict, images_dict, seg_names, save_folder):
    
    """
    Args:
        engine: Ignite Engine, it can be a trainer, validator or evaluator.

    Raises:
        TypeError: When ``output_transform(engine.state.output)[0]`` type is not in
            ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
        TypeError: When ``batch_transform(engine.state.batch)[1]`` type is not in
            ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
        TypeError: When ``output_transform(engine.state.output)`` type is not in
            ``Optional[Union[numpy.ndarray, torch.Tensor]]``.

    """
    # summary_writer.add_summary(summary, global_step).
    step = 1 # self.global_iter_transform(engine.state.iteration)
        #self.bach_transform(engine.state.batch)[0] = self.batch_transform(engine.state.batch)[0]

     
    filename = (
        images_dict["image_meta_dict"]["filename_or_obj"]
        .split("/")[-1]
        .split(".")[0]
    )
    # except:
    #     filename = (
    #         self.bach_transform(engine.state.batch)[0]["saved_meta"]["filename_or_obj"]
    #         .split("/")[-1]
    #         .split(".")[0]
    #     )

    #We will do this for every class. 

    for class_label_name, class_label_val in images_dict["label_names"].items():
        # for 
        pass 

    # IMAGE
    
    if isinstance(show_image, torch.Tensor):
        show_image = show_image.detach().cpu().numpy()
    if show_image is not None:
        if not isinstance(show_image, np.ndarray):
            raise TypeError(
                "show_image must be None or one of "
                f"(numpy.ndarray, torch.Tensor) but is {type(show_image).__name__}."
            )
        plot_2d_or_3d_image(
            # add batch dim and plot the first item
            data=show_image[None],
            step=step,
            writer=self._writer,
            index=0,
            max_channels=self.max_channels,
            max_frames=self.max_frames,
            tag="step_" + str(step) + "_image_" + filename,
        )

    # LABEL
    show_label = self.bach_transform(engine.state.batch)[0]["label"][0, ...][None]
    if isinstance(show_label, torch.Tensor):
        show_label = show_label.detach().cpu().numpy()
    if show_label is not None:
        if not isinstance(show_label, np.ndarray):
            raise TypeError(
                "show_label must be None or one of "
                f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
            )
        plot_2d_or_3d_image(
            # add batch dim and plot the first item
            data=show_label[None],
            step=step,
            writer=self._writer,
            index=0,
            max_channels=self.max_channels,
            max_frames=self.max_frames,
            tag="step_" + str(step) + "_label_" + filename,
        )

    # PREDICTION
    all_preds = self.output_transform(engine.state.output)[0]["pred"]
    for idx in range(all_preds.shape[0]):
        show_prediction = all_preds[idx, ...][None]
        if isinstance(show_prediction, torch.Tensor):
            show_prediction = show_prediction.detach().cpu().numpy()
        if show_prediction is not None:
            if not isinstance(show_prediction, np.ndarray):
                raise TypeError(
                    "show_pred must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
                )
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_prediction[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                max_frames=self.max_frames,
                tag="step_" + str(step) + f"_prediction_for_label_{str(idx)}_" + filename,
            )

    # ALL CLICKS

    #Number of additional input channels:        
    num_classes = len(self.bach_transform(engine.state.batch)[0]["label_names"])
    #Number of channels for the image modality:
    num_intensity_ch = self.bach_transform(engine.state.batch)[0]["image"].shape[0] - 2 * num_classes

    show_pos_clicks = input_tensor[num_intensity_ch:num_intensity_ch + num_classes, ...][None]
    if isinstance(show_pos_clicks, torch.Tensor):
        show_pos_clicks = show_pos_clicks.detach().cpu().numpy()
        # Adding all labels in a single channel tensor
        if show_pos_clicks.shape[1] > 1:
            show_pos_clicks = np.sum(show_pos_clicks, axis=1)
    if show_pos_clicks is not None:
        if not isinstance(show_pos_clicks, np.ndarray):
            raise TypeError(
                "show_pos_clicks must be None or one of "
                f"(numpy.ndarray, torch.Tensor) but is {type(show_pos_clicks).__name__}."
            )
        show_pos_clicks = show_label * (1 - show_pos_clicks)
        plot_2d_or_3d_image(
            # add batch dim and plot the first item
            data=show_pos_clicks[None],
            step=step,
            writer=self._writer,
            index=0,
            max_channels=self.max_channels,
            max_frames=self.max_frames,
            tag="step_" + str(step) + "_all_clicks_" + filename,
        )
    
    #Previous Seg Channels, for each channel we will generate a gif.
    prev_seg_channels = input_tensor[num_intensity_ch + num_classes:, ...]
    for (key_label, val_label) in self.bach_transform(engine.state.batch)[0]["label_names"].items():
        show_prev = prev_seg_channels[val_label - 1, ...][None]
        if isinstance(show_prev, torch.Tensor):
            show_prev = show_prev.detach().cpu().numpy()
        if show_prev is not None:
            if not isinstance(show_prev, np.ndarray):
                raise TypeError(
                    "show_pred must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
                )
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_prev[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                max_frames=self.max_frames,
                tag="step_" + str(step) + f"_prev_seg_for_label_{key_label}_" + filename,
            )

    self._writer.flush()

    # import tensorflow as tf






def metric_extraction(metric_score_configs_path, image_name, metric_type):
    # with open(os.path.join(metric_score_configs_path, 'metric_score_configs_results.csv'), newline='') as f:
        # metric_score_configs_reader = csv.reader(f, delimiter=' ', quotechar='|')
        # first_row = f.readline()
        # first_row = first_row.strip()
        # #print(first_row)
        # #n_cols = first_row.count(',') + 1 
        # metric_score_configss = [[float(j)] if i > 0 else [j] for i,j in enumerate(first_row.split(','))] #[[float(i)] for i in first_row.split(',')]
        # #print(metric_score_configss)
        # for row in metric_score_configs_reader:
        #     row_str_list = row[0].split(',')
        #     #print(row_str_list)
        #     for index, string in enumerate(row_str_list):
        #         if index > 0:
        #             metric_score_configss[index].append(float(string))
        #         elif index == 0:
        #             metric_score_configss[index].append(string)
    spreadsheet_array = pd.read_csv(os.path.join(metric_score_configs_path, f'{metric_type}_results.csv'), sep=',', header=None)
    # np.where(spreadsheet_array[:,0] == image_name)
    metric_score_configs_row = spreadsheet_array.loc[spreadsheet_array[0] == f'{image_name}.nii.gz']
    return metric_score_configs_row.to_numpy()[0]

def plot(input_image, initialisation, guidances, discrepancies_dict, images_dict, seg_names, save_folder, plotting_type, metric_score_configs):
    #If initialisation is deepgrow, then we need to also do the plots for "iter_0_discrepancy"
    
    print('\n')
    print('\n')
    print(plotting_type)
    print('\n')
    print('\n')


    metric_score_configs_path = metric_score_configs[0]
    metric_score_configs_name = metric_score_configs[1] #This denote the subtype of metric which we will refer to it by on our figures.
    metric_score_configs_type = metric_score_configs[2]
    metric_score_multiple_bool= metric_score_configs[3] #This denotes whether the metric is only a single set of scores, or whether they are split
    #across different guidance point set dice scores. I.e, in the case of the deepedit framework, where we had to split into original and guidance per iter.
    image_name = save_folder.split("/")[-2]

    #if the metric scores are split by guidance sets (i.e. point guided) AND/OR because they are from deepedit original and required for that metric, then append to the filepath:
    if eval(metric_score_multiple_bool):
        metric_score_configs_path += f"/{plotting_type}"
    
    metric_score = metric_extraction(metric_score_configs_path, image_name, metric_score_configs_type)[1:]


    #Extracting a variable which determines whether the metric under consideration is an image-by-image metric, or an -image-to-image metric. 

    image_by_image_metric_list = ['Global Dice', "Local Dice"]
    image_to_image_metric_list = ["Temporal Consistency"]

    point_guided_metric_list = ["Local Dice", "Temporal Consistency"]
    non_point_guided_metric_list = ["Global Dice"]

    if metric_score_configs_name not in image_by_image_metric_list and metric_score_configs_name not in image_to_image_metric_list:
        raise Exception(f"Sorry, the {metric_score_configs_name} metric is not supported in the current script.")
    
    if metric_score_configs_name in image_by_image_metric_list:
        image_by_image_bool = True 
        
    else:
        image_by_image_bool = False
        
    if metric_score_configs_name in point_guided_metric_list:
        point_guided_metric_bool = True 

    elif metric_score_configs_name in non_point_guided_metric_list:
        point_guided_metric_bool = False 
    
    if image_by_image_bool and point_guided_metric_bool:
        if initialisation == "autoseg":

            #We create a list from which we will be extracting singletons or pairs depending on the metric type.
            final_metric_list = [('N/A', metric_score[0])] + [(metric_score[i], metric_score[i + 1]) for i in range((metric_score).shape[0] - 1)]

        elif initialisation == "deepgrow":
            #We create a list from which we will be extracting singletons or pairs depending on the metric type.
            final_metric_list = [('N/A', metric_score[0])] + [(metric_score[i], metric_score[i + 1]) for i in range((metric_score).shape[0] - 1)]
    
    elif image_by_image_bool and not point_guided_metric_bool:
        if initialisation == "autoseg":

            final_metric_list = [(metric_score[i], metric_score[i + 1]) for i in range((metric_score.shape[0] - 1))]

        elif initialisation == "deepgrow":

            final_metric_list = [('N/A', metric_score[0])] + [(metric_score[i], metric_score[i + 1]) for i in range((metric_score).shape[0] - 1)]

    else:
        final_metric_list = metric_score.tolist()

    discrepancies_iters_names_list = list(discrepancies_dict.keys())
    if initialisation == 'deepgrow':

        #Generating the list of names for the 


    
        for index, (iter, guidance_point_set) in enumerate(guidances.items()):
            #This guidance point set variable will be a dict for each class with their respective guidance points

            # Loading in the discrepancies for the current iteration under consideration: 
            discrepancy_set = [discrepancies_dict[discrepancies_iters_names_list[index]], discrepancies_dict[discrepancies_iters_names_list[index + 1]]]
            


            if index == 0:
                #If index=0 i.e. initialisation, then we can just use a zeros matrix
                # prior_seg = np.zeros(input_image.shape)
                current_seg = copy.deepcopy(images_dict[seg_names[index]][0])
            else:
                #In this circumstance we can proceed as normal.
                # Loading in the prior segmentation for the current iteration:
                prior_seg = copy.deepcopy(images_dict[seg_names[index - 1]][0].array)
                # Loading in the updated segmentation for the current iteration:
                current_seg = copy.deepcopy(images_dict[seg_names[index]][0].array)

            current_gt = copy.deepcopy(images_dict["GT"][0].array) 
            
            #we need to then split the GT into its constituent classes for plotting:
            split_gt = dict()
            for (class_name, class_val) in images_dict["labels"].items():
                tmp_label = np.where(current_gt == class_val, 1, 0)
                #tmp_label[current_gt == class_val] = 1
                split_gt[class_name] = tmp_label

                
            #Here we plot the points for each classes' guidance points aligned with discrepancies etc, by taking a 2D slice according to the guidance point location.

            for (class_name, guidance) in guidance_point_set.items():
                
                #Creating the save folder for this set of points:
                current_iter_class_save_folder = os.path.join(save_folder, iter, class_name) #TODO: when changing this for the only_current_set_points
                

                #Extracting the corresponding discrepancy maps:
                current_class_discrepancies = [copy.deepcopy(discrepancy_set[0][class_name]), copy.deepcopy(discrepancy_set[1][class_name])]#.array

                #Extracting the corresponding GT map for the current class
                current_class_gt_label = copy.deepcopy(split_gt[class_name])


                #Extracting the corresponding previous and current seg map for the current class. We will treat everything else as background.
                
                # if index == 0 and class_name == "background":
                    # prior_seg_slice = 1 - prior_seg#[axis_index, :, :]
                # else:
                if index == 0:
                    current_class_prior_seg = np.zeros(input_image.shape)
                else:
                    current_class_prior_seg = np.where(prior_seg == images_dict["label_names"][class_name], 1, 0)  #[axis_index, :, :]

                current_class_current_seg = np.where(current_seg == images_dict["label_names"][class_name], 1, 0)


                #For each guidance point we add on a blob representing the click. Then we take a 2D slice for each point and plot that slice.

                #We are going to convert these images that we want to plot into RGB.. so that we can represent the points with a distinct colour.
                


                #Lets just create a 4D tensor: 2 x HWD. The first channel is just the point click tensor. Second
                #channel is just the original image under consideration. We then convert the 4D tensor into an RGB image by making the slice that we want
                # the M x N array in the RGB image. The RGB channels are populated using the slice from the guidance blob tensor, and the original image tensor.




                guidance_tensor = np.zeros(input_image.shape)
                for point in guidance:
                        
                    if point == []:
                        continue 
                    guidance_tensor[point[0], point[1], point[2]] = 1

                    
                    #current_discrepancy[point[0], point[1], point[2]] = 


                
                for point_index, point in enumerate(guidance): 
                    if point == []:
                        continue 
                    print(f'Class {class_name}: Iter {iter}. Point {point_index}')
                    print(point)
                    #print(f'Class val is {class_val}')
                    print(f'prior seg value here is {current_class_prior_seg[point[0], point[1], point[2]]}')
                    current_iter_class_point_save_folder = os.path.join(current_iter_class_save_folder, f'point_{point_index + 1}')
                    os.makedirs(current_iter_class_point_save_folder, exist_ok=True)


                    #We will take the R-L axis as the one to take a slice from.
                    axis_index = point[0]

                    prior_false_neg_discrepancy_slice = current_class_discrepancies[0][0][0, axis_index, :, :]
                    prior_false_pos_discrepancy_slice = current_class_discrepancies[0][1][0, axis_index, :, :]
                    #gt_slice = current_gt[axis_index, :, :]
                    updated_false_neg_discrepancy_slice = current_class_discrepancies[1][0][0, axis_index, :, :]
                    updated_false_pos_discrepancy_slice = current_class_discrepancies[1][1][0, axis_index, :, :]

                    image_slice = input_image[axis_index, :, :]
                    
                    #If the previous seg is "initialisation", then we will assign prior seg to be
                    # if index == 0 and class_name == "background":
                    #     prior_seg_slice = 1 - prior_seg[axis_index, :, :]
                    # else:
                    # prior_seg_slice = prior_seg[axis_index, :, :]
                    
                    # guidance_slice = guidance_tensor[axis_index, :, :]
                    current_gt_slice = current_class_gt_label[axis_index, :, :]

                    prior_true_pos_slice = current_class_prior_seg[axis_index, :, :] * current_gt_slice 
                    current_true_pos_slice = current_class_current_seg[axis_index, :, :] * current_gt_slice

                    #Stacking the True positive, False negative and false positive channels into one RGB image: 
                    rgb_prior_seg = np.stack([prior_false_neg_discrepancy_slice, prior_true_pos_slice, prior_false_pos_discrepancy_slice], axis=2, dtype=np.float32)
                    rgb_current_seg = np.stack([updated_false_neg_discrepancy_slice, current_true_pos_slice, updated_false_pos_discrepancy_slice], axis=2, dtype=np.float32)

                    #Stacking the gt and image:

                    rgb_gt = np.stack([current_gt_slice, current_gt_slice, current_gt_slice], axis=2, dtype=np.float32)
                    rgb_image = np.stack([image_slice, image_slice, image_slice], axis=2, dtype=np.float32)


                    #Collating all of these variables into one list and then iterating through them:
                    tensor_list = [rgb_prior_seg, rgb_current_seg, rgb_gt, rgb_image]
                    tensor_list_var_name = [f'Segment: Iteration {index - 1}', f'Segment: Iteration {index}',  'Ground Truth Slice', 'Image Slice']
                    
                    
                    #Extracting the metric scores for the current iteration:
                    
                    metric_scores_current_iter = final_metric_list[index]

                    ################# Saving plots with the guidance points on them ##################################
                    fig, ax = plt.subplots(2, 2, figsize=(image_slice.shape[1]/6, image_slice.shape[0]/6)) 
                    red_patch = mpatches.Patch(facecolor=(1,0,0), label='False Negative Region', edgecolor="black")
                    green_patch = mpatches.Patch(facecolor=(0,1,0), label='True Positive Region', edgecolor="black")
                    blue_patch = mpatches.Patch(facecolor=(0,0,1), label='False Positive Region', edgecolor="black")
                    cyan_patch = mpatches.Patch(facecolor='cyan', label='Guidance Point', edgecolor="black")
                    white_patch = mpatches.Patch(facecolor="white", label=f'{class_name.capitalize()} GT Region', edgecolor="black")
                    black_patch = mpatches.Patch(facecolor="black", label=f'Non-{class_name.capitalize()} GT Region', edgecolor="black")
                    fig.legend(bbox_to_anchor=(0,0.9, 1, 0.2), mode="expand", loc="lower left", ncol=6, handles=[red_patch, green_patch, blue_patch, cyan_patch, white_patch, black_patch]).get_frame().set_edgecolor("black")          


                    if image_by_image_bool:
                        fig.suptitle(f'Segmentation Evolution - Metric: {metric_score_configs_name}, Class: {class_name.capitalize()}, Point: {point_index + 1}')

                        if metric_scores_current_iter[0] == "N/A":
                            secondary_titles = [f'Metric: N/A', f'Metric: {metric_scores_current_iter[1]:.5f}', None, None]
                        else:
                            secondary_titles = [f'Metric: {metric_scores_current_iter[0]:.5f}', f'Metric: {metric_scores_current_iter[1]:.5f}', None, None]
                    else:
                        fig.suptitle(f"Segmentation Evolution - Metric: {metric_score_configs_name}: {metric_scores_current_iter}, Class: {class_name.capitalize()}, Point: {point_index + 1}")
                        secondary_titles = [None, None, None, None]


                    for i, plot_type in enumerate(tensor_list):
            
                        try: 
                            plot_hor_ax = i // 2 
                        except:
                            #If i = 0 then it would be undefined. But if i = 0 then it is just the first row anyways
                            plot_hor_ax = 0
                        if plot_hor_ax > 0:
                            plot_vert_ax = i - plot_hor_ax * 2
                        else:
                            #If first row then it would be undefined. 
                            plot_vert_ax = i 
                        

                        ax[plot_hor_ax, plot_vert_ax].imshow(plot_type, aspect='auto')
                        
                        if not tensor_list_var_name[i] == f'point_{point_index + 1}_gt_slice':

                            for j in range(np.where(guidance_tensor[axis_index, :, :] == 1)[0].shape[0]):
                                
                                y = np.where(guidance_tensor[axis_index, :, :] == 1)[0][j]
                                x = np.where(guidance_tensor[axis_index, :, :] == 1)[1][j]

                                ax[plot_hor_ax, plot_vert_ax].scatter(x, y, marker='o', s=50, edgecolors='black', color=(0/255,255/255,255/255))

                        
                        if secondary_titles[i] is not None:
                            ax[plot_hor_ax, plot_vert_ax].set_title(tensor_list_var_name[i], loc="left")
                        # if secondary_titles[i] is not None:
                            ax[plot_hor_ax, plot_vert_ax].set_title(f"{secondary_titles[i]}", loc="right") 
                        else:
                            ax[plot_hor_ax, plot_vert_ax].set_title(tensor_list_var_name[i])                

                    plt.savefig(os.path.join(current_iter_class_point_save_folder,f'seg_and_{metric_score_configs_name}.png'))
                    plt.close()
            
        #Now plotting the "guidances per iter" ones.
        # for guidance_point_set in guidances_per_iter:


    else:

        #For autoseg initialisations, we do not need to start from discrepancy_iter_0.
        for index, (iter, guidance_point_set) in enumerate(guidances.items()):
            #This guidance point set variable will be a dict for each class with their respective guidance points
            
            # Loading in the discrepancies for the current iteration under consideration: 
            
            # Loading in the discrepancies for the current iteration under consideration: 
            # discrepancy_set = discrepancies_dict[iter]
            discrepancy_set = [discrepancies_dict[discrepancies_iters_names_list[index]], discrepancies_dict[discrepancies_iters_names_list[index + 1]]]
            


            # if index == 0:
            #     #If index=0 i.e. initialisation, then we can just use a zeros matrix
            #     prior_seg = np.zeros(input_image.shape)
            #     current_seg = copy.deepcopy(images_dict[seg_names[index]][0])
            # else:
                #In this circumstance we can proceed as normal.
                # Loading in the prior segmentation for the current iteration:
            prior_seg = copy.deepcopy(images_dict[seg_names[index]][0].array)
            # Loading in the updated segmentation for the current iteration:
            current_seg = copy.deepcopy(images_dict[seg_names[index + 1]][0].array)

            current_gt = copy.deepcopy(images_dict["GT"][0].array) 
            
            #we need to then split the GT into its constituent classes for plotting:
            split_gt = dict()
            for (class_name, class_val) in images_dict["labels"].items():
                tmp_label = np.where(current_gt == class_val, 1, 0)
                #tmp_label[current_gt == class_val] = 1
                split_gt[class_name] = tmp_label

            
            #Here we plot the points for each classes' guidance points aligned with discrepancies etc, by taking a 2D slice according to the guidance point location.

            for (class_name, guidance) in guidance_point_set.items():
                
                #Creating the save folder for this set of points:
                current_iter_class_save_folder = os.path.join(save_folder, iter, class_name) #TODO: when changing this for the only_current_set_points
                

                #Extracting the corresponding discrepancy map:
                # current_class_discrepancy = copy.deepcopy(discrepancy_set[class_name])#.array
                #Extracting the corresponding discrepancy maps:
                current_class_discrepancies = [copy.deepcopy(discrepancy_set[0][class_name]), copy.deepcopy(discrepancy_set[1][class_name])]#.array

                #Extracting the corresponding GT map for the current class
                current_class_gt_label = copy.deepcopy(split_gt[class_name])


                #Extracting the corresponding previous and current seg map for the current class. We will treat everything else as background.
                
                # if index == 0 and class_name == "background":
                    # prior_seg_slice = 1 - prior_seg#[axis_index, :, :]
                # else:
                current_class_prior_seg = np.where(prior_seg == images_dict["label_names"][class_name], 1, 0)  #[axis_index, :, :]

                current_class_current_seg = np.where(current_seg == images_dict["label_names"][class_name], 1, 0)


                
                #For each guidance point we add on a blob representing the click. Then we take a 2D slice for each point and plot that slice.

                #We are going to convert these images that we want to plot into RGB.. so that we can represent the points with a distinct colour.
                


                #Lets just create a 4D tensor: 2 x HWD. The first channel is just the point click tensor. Second
                #channel is just the original image under consideration. We then convert the 4D tensor into an RGB image by making the slice that we want
                # the M x N array in the RGB image. The RGB channels are populated using the slice from the guidance blob tensor, and the original image tensor.




                guidance_tensor = np.zeros(input_image.shape)
                for point in guidance:
                    if point == []:
                        continue
                    guidance_tensor[point[0], point[1], point[2]] = 1

                    
                    #current_discrepancy[point[0], point[1], point[2]] = 


                
                for point_index, point in enumerate(guidance):
                    if point == []:
                        continue
                    print(f'Class {class_name}: Iter {iter}. Point {point_index}')
                    print(point)
                    #print(f'Class val is {class_val}')
                    print(f'prior seg value here is {current_class_prior_seg[point[0], point[1], point[2]]}')
                    current_iter_class_point_save_folder = os.path.join(current_iter_class_save_folder, f'point_{point_index + 1}')
                    os.makedirs(current_iter_class_point_save_folder, exist_ok=True)


                    #We will take the R-L axis as the one to take a slice from.
                    axis_index = point[0]

                    prior_false_neg_discrepancy_slice = current_class_discrepancies[0][0][0, axis_index, :, :]
                    prior_false_pos_discrepancy_slice = current_class_discrepancies[0][1][0, axis_index, :, :]
                    #gt_slice = current_gt[axis_index, :, :]
                    updated_false_neg_discrepancy_slice = current_class_discrepancies[1][0][0, axis_index, :, :]
                    updated_false_pos_discrepancy_slice = current_class_discrepancies[1][1][0, axis_index, :, :]
                    
                    image_slice = input_image[axis_index, :, :]
                    
                    #If the previous seg is "initialisation", then we will assign prior seg to be
                    # if index == 0 and class_name == "background":
                    #     prior_seg_slice = 1 - prior_seg[axis_index, :, :]
                    # else:
                    # prior_seg_slice = prior_seg[axis_index, :, :]
                    # prior_seg_slice = current_class_prior_seg[axis_index, :, :]
                    # current_seg_slice = current_class_current_seg[axis_index, :, :]
                    # guidance_slice = guidance_tensor[axis_index, :, :]
                    current_gt_slice = current_class_gt_label[axis_index, :, :]

                    prior_true_pos_slice = current_class_prior_seg[axis_index, :, :] * current_gt_slice 
                    current_true_pos_slice = current_class_current_seg[axis_index, :, :] * current_gt_slice

                    #Stacking the True positive, False negative and false positive channels into one RGB image: 
                    rgb_prior_seg = np.stack([prior_false_neg_discrepancy_slice, prior_true_pos_slice , prior_false_pos_discrepancy_slice], axis=2, dtype=np.float32)
                    rgb_current_seg = np.stack([updated_false_neg_discrepancy_slice, current_true_pos_slice, updated_false_pos_discrepancy_slice], axis=2, dtype=np.float32)

                    #Stacking the gt and image:

                    rgb_gt = np.stack([current_gt_slice, current_gt_slice, current_gt_slice], axis=2, dtype=np.float32)
                    rgb_image = np.stack([image_slice, image_slice, image_slice], axis=2, dtype=np.float32)


                    #Collating all of these variables into one list and then iterating through them:
                    tensor_list = [rgb_prior_seg, rgb_current_seg, rgb_gt, rgb_image]
                    tensor_list_var_name = [f'Segment: Iteration {index}', f'Segment: Iteration {index + 1}',  'Ground Truth Slice', 'Image Slice']
                
                    #Extracting the metric scores for the current iteration:
                    
                    metric_scores_current_iter = final_metric_list[index]

                    ################# Saving plots with the guidance points on them ##################################
                    fig, ax = plt.subplots(2, 2, figsize=(image_slice.shape[1]/6, image_slice.shape[0]/6)) #, figsize=(30,20))

                    red_patch = mpatches.Patch(facecolor=(1,0,0), label='False Negative Region', edgecolor="black")
                    green_patch = mpatches.Patch(facecolor=(0,1,0), label='True Positive Region', edgecolor="black")
                    blue_patch = mpatches.Patch(facecolor=(0,0,1), label='False Positive Region', edgecolor="black")
                    cyan_patch = mpatches.Patch(facecolor='cyan', label='Guidance Point', edgecolor="black")
                    white_patch = mpatches.Patch(facecolor="white", label=f'{class_name.capitalize()} GT Region', edgecolor="black")
                    black_patch = mpatches.Patch(facecolor="black", label=f'Non-{class_name.capitalize()} GT Region', edgecolor="black")
                    fig.legend(bbox_to_anchor=(0,0.9, 1, 0.2), mode="expand", loc="lower left", ncol=6, handles=[red_patch, green_patch, blue_patch, cyan_patch, white_patch, black_patch]).get_frame().set_edgecolor("black")          

                    if image_by_image_bool:
                        fig.suptitle(f'Segmentation Evolution - Metric: {metric_score_configs_name}, Class: {class_name.capitalize()}, Point: {point_index + 1}')

                        if metric_scores_current_iter[0] == "N/A":
                            secondary_titles = [f'Metric: N/A', f'Metric: {metric_scores_current_iter[1]:.5f}', None, None]
                        else:
                            secondary_titles = [f'Metric: {metric_scores_current_iter[0]:.5f}', f'Metric: {metric_scores_current_iter[1]:.5f}', None, None]
                    else:
                        fig.suptitle(f"Segmentation Evolution - Metric: {metric_score_configs_name}: {metric_scores_current_iter}, Class: {class_name.capitalize()}, Point: {point_index + 1}")
                        secondary_titles = [None, None, None, None]


                    for i, plot_type in enumerate(tensor_list):
            
                        try: 
                            plot_hor_ax = i // 2 
                        except:
                            #If i = 0 then it would be undefined. But if i = 0 then it is just the first row anyways
                            plot_hor_ax = 0
                        if plot_hor_ax > 0:
                            plot_vert_ax = i - plot_hor_ax * 2
                        else:
                            #If first row then it would be undefined. 
                            plot_vert_ax = i 
                        

                        ax[plot_hor_ax, plot_vert_ax].imshow(plot_type, aspect='auto')
                        
                        if not tensor_list_var_name[i] == f'point_{point_index + 1}_gt_slice':

                            for j in range(np.where(guidance_tensor[axis_index, :, :] == 1)[0].shape[0]):
                                
                                y = np.where(guidance_tensor[axis_index, :, :] == 1)[0][j]
                                x = np.where(guidance_tensor[axis_index, :, :] == 1)[1][j]

                                ax[plot_hor_ax, plot_vert_ax].scatter(x, y, marker='o', s=50, edgecolors='black', color=(0/255,255/255,255/255))

                        
                        if secondary_titles[i] is not None:
                            ax[plot_hor_ax, plot_vert_ax].set_title(tensor_list_var_name[i], loc="left")
                        # if secondary_titles[i] is not None:
                            ax[plot_hor_ax, plot_vert_ax].set_title(f"{secondary_titles[i]}", loc="right") 
                        else:
                            ax[plot_hor_ax, plot_vert_ax].set_title(tensor_list_var_name[i])              

                    plt.savefig(os.path.join(current_iter_class_point_save_folder,f'seg_and_{metric_score_configs_name}.png'))
                    plt.close()
    return 

def plotting_images(save_folder, seg_names, images_dict, discrepancies_dict, guidance_points, guidances_per_iter, framework, initialisation, output_type, metric_score_configs_configs):
    '''
    Function which plots the guidance points onto the appropriate images etc, according to the approximate slices which they are occuring on.
    seg_names: The images_dict keys for the segmentations which we will be plotting.
    images_dict: Dict containing the segmentations across all the iterations, plus the original input image.
    discrepancies_dict: Dict containing the discrepancies across all the relevant iterations.
    guidance_points: The guidance points for the input request
    guidances_per_iter: ONLY the new guidance points for the current iteration (this is the same as guidance_points for deepedit++ but NOT for deepedit original)
    framework: the framework for which we are plotting, if deepedit then we need separate plots for thhhe two guidance arguments.
    initialisation: this variable states whether the initial segmentation is a deepgrow seg., this is because the discrepancies will be denoted as iter_0!
    output_type: this variable dictates whether it is being saved as a png type output, or for the tensorboard display!
    '''

    if output_type == "png":

        input_image = images_dict["image"][0].array
        input_image /= np.max(np.abs(input_image))

        if framework == 'deepedit':
            plot(input_image, initialisation, guidance_points, discrepancies_dict, images_dict, seg_names, os.path.join(save_folder, 'original_guidance'), 'original_guidance', metric_score_configs_configs)
            
        # elif framework == 'deepeditplusplus':
        #     #Only need to plot one of them.
        #     pass



        if framework == 'deepedit':
            #Then plot both sets of guidance points.

            plot(input_image, initialisation, guidances_per_iter, discrepancies_dict, images_dict, seg_names, os.path.join(save_folder, 'guidance_per_iter'), 'guidance_per_iter', metric_score_configs_configs)
            
                
            
        if framework == 'deepeditplusplus':
            #Only need to plot one of them. Which is the set of points per iteration.
            plot(input_image, initialisation, guidances_per_iter, discrepancies_dict, images_dict, seg_names, os.path.join(save_folder, 'guidance_per_iter'), None, metric_score_configs_configs)
    
    elif output_type == "tensorboard":
        input_image = images_dict["image"][0].array
        input_image /= np.max(np.abs(input_image))

        if framework == "deepedit":
            plot_tensorboard(input_image, initialisation, guidance_points, discrepancies_dict, images_dict, seg_names, save_folder)
        elif framework == "deepeditplusplus":

            plot_tensorboard(input_image, initialisation, guidances_per_iter, discrepancies_dict, images_dict, seg_names, save_folder)

    # return 

def plotting_func(image_config, task_configs, label_config_path, save_folder):
    ''' 
        in the image config, element 1 is the directory path for the task & checkpoint/model config combination
        element 2 is the name of the image

        task configs is the config for the task that was being performed, helps us structure the code for extracting the relevant images, element 1 is the framework name, 2
        element 2 is the task name, if element 2 is deepedit then the remaining two elements are the initialisation and the number of iterations of clicking interactions.    

        label_config_path is the path to the text file containing the label mappings etc for the current task (e.g. binary segmentation for BRATS)

        save_folder is the path to the model & framework configuration for which we are analysing the segmentation outputs. 
        

    '''
        ############ If editing ###############
    if task_configs[1] == "deepedit":
        input_image = [os.path.join(image_config[0], image_config[1])]
        initialisation_path = [os.path.join(image_config[0], 'labels', task_configs[2], image_config[1])]
        deepedit_paths = [os.path.join(image_config[0], 'labels', f'deepedit_iteration_{i}', image_config[1]) for i in range(1, int(task_configs[3]))]
        final_path = [os.path.join(image_config[0], 'labels', 'final', image_config[1])]

        all_segmentation_paths = input_image + initialisation_path + deepedit_paths + final_path 
        original_gt = os.path.join(image_config[0], 'labels', 'original', image_config[1])

        transform_input_dict_keys = ['image'] + [task_configs[2]] + [f'deepedit_iteration_{i}' for i in range(1, int(task_configs[3]))] + ['final']
        #Saving a list with a set of the names for the segmentations
        seg_names = [task_configs[2]] + [f'deepedit_iteration_{i}' for i in range(1, int(task_configs[3]))] + ['final']
        #Here we fuse together the lists of the segmentation (and input image) filepaths and their corresponding key_names for the key-val pairs of our input dictionaries.

        transform_input_dict = dict() 

        for (key, path) in zip(transform_input_dict_keys, all_segmentation_paths):
            transform_input_dict[key] = path 

        #Adding the ground truth path:
        transform_input_dict["GT"] = original_gt 


        

        #We obtain the inference outputs and the original GT in their original orientation! The guidance points are generated in RAS 

        #For each inference output, we compute the disrepancy in the RAS orientation. 

        #TODO: implement the function for this!


        #This function loads in all the images once (means we do not have to load the same image in multiple times)
        
        loading_transforms_output_dictionary = loading_images(transform_input_dict_keys, label_config_path, transform_input_dict)


        
        # Here we are generating the discrepancies across the iterations, obviously not required for the final iteration since there are no further interactions.
        # The outputs of the discrepancies are K classes x 2 (false negative: index 0, and false positive: index 1)
        if task_configs[2] == "autoseg":

            #In this case we do not want to obtain the discrepancy prior to the initialisation because it does not have any clicks simulated
            discrepancies_output_dictionary = generate_discrepancies(loading_transforms_output_dictionary, transform_input_dict_keys[1:])

        else: #i.e. for deepgrow

            #In this case we DO want to obtain the discrepancy prior to the initialisation (but this is just the GT!)

            #We generate the discrepancies for the initialisation onwards, and we use the GT as the discrepancy for iteration 0.
            discrepancies_output_dictionary_tmp = generate_discrepancies(loading_transforms_output_dictionary, transform_input_dict_keys[1:])

            original_discrepancy = dict()
            GT_copy = copy.deepcopy(loading_transforms_output_dictionary["GT"])

            for (class_name, class_val) in loading_transforms_output_dictionary['label_names'].items():
                print((class_name, class_val))
                #create a list to store the false-neg "discrepancy". Here this is just the GT for each class for deepgrow.
                #tmp_label = np.zeros(GT_copy.shape)
                false_neg = np.where(GT_copy == class_val, 1, 0)
                false_pos = 1 - false_neg
                original_discrepancy[class_name] = [false_neg, false_pos]

            discrepancies_output_dictionary = dict()
            discrepancies_output_dictionary["discrepancy_iter_0"] = copy.deepcopy(original_discrepancy)
            
            #We want the discrepancies and their key:val pairs to be in iteration order.
            for key in discrepancies_output_dictionary_tmp.keys():
                discrepancies_output_dictionary[key] = discrepancies_output_dictionary_tmp[key]

            # discrepancies_output_dictionary["discrepancy_iter_0"] = copy.deepcopy(loading_transforms_output_dictionary["GT"])

        #Loading in the guidance points:
        #guidance_points_paths = []
        target_image_split = image_config[1].split('.')
        target_image = target_image_split[0]
        if task_configs[2] == "autoseg":

            # initialisation_path = [os.path.join(image_config[0], 'labels', 'guidance_points', task_configs[2], image_config[1])]
            deepedit_guidance_paths = [os.path.join(image_config[0], 'labels', 'guidance_points', f'deepedit_iteration_{i}.json') for i in range(1, int(task_configs[3]))]
            final_guidance_path = [os.path.join(image_config[0], 'labels', 'guidance_points', 'final_iteration.json')]

            all_guidance_paths = deepedit_guidance_paths + final_guidance_path


            #TODO: Load the guidance points + add the function which will plot the guidance points and discrepancies...... These are saved in chronological order.
            guidance_points = []
            for guidance_path in all_guidance_paths:
                with open(guidance_path, 'r') as f:
                    saved_dict = json.load(f)
                    guidance_points.append(saved_dict[target_image])
            
            #If framework is original deepedit, then we need to also obtain the "new" clicks separately for each iteration so we can also plot them with the discrepancies.
            if task_configs[0] == "deepeditplusplus":
                #Converting the guidance_points list into a searchable dict with the keys being the corresponding discrepancy_iterations.
                guidance_points = [{f'discrepancy_iter_{i + 1}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
                guidances_per_iter = copy.deepcopy(guidance_points) 
                
            elif task_configs[0] == "deepedit":
                guidances_per_iter = [guidance_points[0]]

                for i in range(1, len(guidance_points)):
                    new_guidance = dict()
                    for class_label in guidance_points[i].keys():
                        #Computing the new guidance points for each class label 
                        new_guidance_class_label = [point for point in guidance_points[i][class_label] if point not in guidance_points[i-1][class_label]]
                        new_guidance[class_label] = new_guidance_class_label
                    #Appending the new guidance points to the list of "new guidances" per iteration.
                    guidances_per_iter.append(new_guidance)

                guidance_points = [{f'discrepancy_iter_{i + 1}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
                guidances_per_iter = [{f'discrepancy_iter_{i + 1}':guidance_set} for i, guidance_set in enumerate(guidances_per_iter)]
        
        else:
            initialisation_guidance_path = [os.path.join(image_config[0], 'labels', 'guidance_points', task_configs[2] + '.json')]
            deepedit_guidance_paths = [os.path.join(image_config[0], 'labels', 'guidance_points', f'deepedit_iteration_{i}.json') for i in range(1, int(task_configs[3]))]
            final_guidance_path = [os.path.join(image_config[0], 'labels', 'guidance_points', 'final_iteration.json')]

            all_guidance_paths = initialisation_guidance_path + deepedit_guidance_paths + final_guidance_path 


            #TODO: Load the guidance points + add the function which will plot the guidance points and discrepancies...... These are saved in chronological order.
            guidance_points = []
            for guidance_path in all_guidance_paths:
                with open(guidance_path, 'r') as f:
                    saved_dict = json.load(f)
                    guidance_points.append(saved_dict[target_image])
            
            #If framework is original deepedit, then we need to also obtain the "new" clicks separately for each iteration so we can also plot them with the discrepancies.
            if task_configs[0] == "deepeditplusplus":
                #Converting the guidance_points list into a searchable dict with the keys being the corresponding discrepancy_iterations.
                guidance_points = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
                guidances_per_iter = copy.deepcopy(guidance_points) 
                
            elif task_configs[0] == "deepedit":
                guidances_per_iter = [guidance_points[0]]

                for i in range(1, len(guidance_points)):
                    new_guidance = dict()
                    for class_label in guidance_points[i].keys():
                        #Computing the new guidance points for each class label 
                        new_guidance_class_label = [point for point in guidance_points[i][class_label] if point not in guidance_points[i-1][class_label]]
                        new_guidance[class_label] = new_guidance_class_label
                    #Appending the new guidance points to the list of "new guidances" per iteration.
                    guidances_per_iter.append(new_guidance)
                
                guidance_points = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
                guidances_per_iter = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidances_per_iter)]


        guidance_points_plotting = dict()
        for point in guidance_points: 
            guidance_points_plotting = guidance_points_plotting | point

        guidances_per_iter_plotting = dict()
        for point in guidances_per_iter: 
            guidances_per_iter_plotting = guidances_per_iter_plotting | point 

        #guidance_points

        
        output_save_folder = os.path.join(save_folder, target_image)
        if os.path.exists(output_save_folder):
            shutil.rmtree(output_save_folder)
        

        return output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, task_configs[0],  initialisation
        # plotting_images(output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, task_configs[0],  initialisation)

    ######## If only deepgrow, then just load the GT in as the "discrepancy", list of segmentations would just be "final" ############
    elif task_configs[1] == "deepgrow":
        input_image = [os.path.join(image_config[0], image_config[1])]
        # initialisation_path = [os.path.join(image_config[0], 'labels', task_configs[2], image_config[1])]
        # deepedit_paths = [os.path.join(image_config[0], 'labels', f'deepedit_iteration_{i}', image_config[1]) for i in range(1, int(task_configs[3]))]
        final_path = [os.path.join(image_config[0], 'labels', 'final', image_config[1])]

        all_segmentation_paths = input_image + final_path 
        original_gt = os.path.join(image_config[0], 'labels', 'original', image_config[1])

        transform_input_dict_keys = ['image'] + ['final']
        #Saving a list with a set of the names for the segmentations
        seg_names = ['final']
        #Here we fuse together the lists of the segmentation (and input image) filepaths and their corresponding key_names for the key-val pairs of our input dictionaries.

        transform_input_dict = dict() 

        for (key, path) in zip(transform_input_dict_keys, all_segmentation_paths):
            transform_input_dict[key] = path 

        #Adding the ground truth path:
        transform_input_dict["GT"] = original_gt 


        

        #We obtain the inference outputs and the original GT in their original orientation! The guidance points are generated in RAS 

        #For each inference output, we compute the disrepancy in the RAS orientation. 

        #TODO: implement the function for this!


        #This function loads in all the images once (means we do not have to load the same image in multiple times)
        
        loading_transforms_output_dictionary = loading_images(transform_input_dict_keys, label_config_path, transform_input_dict)




            #In this case we DO want to obtain the discrepancy prior to the initialisation (but this is just the GT!)

        #We generate the discrepancies for the initialisation onwards, and we use the GT as the discrepancy for iteration 0.
        #discrepancies_output_dictionary = generate_discrepancies(loading_transforms_output_dictionary, transform_input_dict_keys[1:-1])

        original_discrepancy = dict()
        GT_copy = copy.deepcopy(loading_transforms_output_dictionary["GT"])

        for (class_name, class_val) in loading_transforms_output_dictionary['label_names'].items():
            print((class_name, class_val))
            #create a list to store the false-neg "discrepancy". Here this is just the GT for each class for deepgrow.
            #tmp_label = np.zeros(GT_copy.shape)
            false_neg = np.where(GT_copy == class_val, 1, 0)
            false_pos = 1 - false_neg
            original_discrepancy[class_name] = [false_neg, false_pos]



        discrepancies_output_dictionary_tmp = generate_discrepancies(loading_transforms_output_dictionary, transform_input_dict_keys[1:])
        discrepancies_output_dictionary = dict()

        discrepancies_output_dictionary["discrepancy_iter_0"] = copy.deepcopy(original_discrepancy)
        discrepancies_output_dictionary["discrepancy_iter_1"] = copy.deepcopy(discrepancies_output_dictionary_tmp["discrepancy_iter_1"])
        # discrepancies_output_dictionary["discrepancy_iter_0"] = copy.deepcopy(loading_transforms_output_dictionary["GT"])

        #Loading in the guidance points:
        #guidance_points_paths = []
        target_image_split = image_config[1].split('.')
        target_image = target_image_split[0]
        
        
        #initialisation_guidance_path = [os.path.join(image_config[0], 'labels', 'guidance_points', task_configs[2] + '.json')]
        #deepedit_guidance_paths = [os.path.join(image_config[0], 'labels', 'guidance_points', f'deepedit_iteration_{i}.json') for i in range(1, int(task_configs[3]))]
        final_guidance_path = [os.path.join(image_config[0], 'labels', 'guidance_points', 'final_iteration.json')]

        all_guidance_paths = final_guidance_path 


        #TODO: Load the guidance points + add the function which will plot the guidance points and discrepancies...... These are saved in chronological order.
        guidance_points = []
        for guidance_path in all_guidance_paths:
            with open(guidance_path, 'r') as f:
                saved_dict = json.load(f)
                guidance_points.append(saved_dict[target_image])
        
        #If framework is original deepedit, then we need to also obtain the "new" clicks separately for each iteration so we can also plot them with the discrepancies.
        if task_configs[0] == "deepeditplusplus":
            #Converting the guidance_points list into a searchable dict with the keys being the corresponding discrepancy_iterations.
            guidance_points = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
            guidances_per_iter = copy.deepcopy(guidance_points) 
            
        elif task_configs[0] == "deepedit":
            guidances_per_iter = [guidance_points[0]]

            for i in range(1, len(guidance_points)):
                new_guidance = dict()
                for class_label in guidance_points[i].keys():
                    #Computing the new guidance points for each class label 
                    new_guidance_class_label = [point for point in guidance_points[i][class_label] if point not in guidance_points[i-1][class_label]]
                    new_guidance[class_label] = new_guidance_class_label
                #Appending the new guidance points to the list of "new guidances" per iteration.
                guidances_per_iter.append(new_guidance)
            
            guidance_points = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidance_points)]
            guidances_per_iter = [{f'discrepancy_iter_{i}':guidance_set} for i, guidance_set in enumerate(guidances_per_iter)]

        guidance_points_plotting = dict()
        for point in guidance_points: 
            guidance_points_plotting = guidance_points_plotting | point

        guidances_per_iter_plotting = dict()
        for point in guidances_per_iter: 
            guidances_per_iter_plotting = guidances_per_iter_plotting | point 

        #guidance_points

        
        output_save_folder = os.path.join(save_folder, target_image)
        
        return output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, task_configs[0],  'deepgrow'
        # plotting_images(output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, task_configs[0],  'deepgrow')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/imagesTs")
    parser.add_argument("--datetime", nargs="+", default=["31052024_195641"])#["02062024_144027"]) #["31052024_195641"])
    parser.add_argument("--checkpoint", nargs="+", default=["best_val_score_epoch"])
    parser.add_argument("--infer_run", default="0")
    parser.add_argument("-ta", "--task", nargs="+", default=["deepeditplusplus", "deepgrow"], help="The framework selection + subtask/mode which we want to execute")
    parser.add_argument("--app_dir", default = "MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus")
    parser.add_argument("--image_names", nargs='+', default=["BraTS2021_01620.nii.gz"])
    parser.add_argument("--label_configs", default="BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_label_configs.txt", help="Label mapping configs for the task under consideration")
    parser.add_argument("--output_type", default="png")
    parser.add_argument("--metric_config", nargs="+", default=["results", "Global Dice", "dice_score", "False"], help='First arg is the folder which the metrics are saved, second arg is the name of the metric, third arg is the metric type (or just "score"), fourth denotes whether there is just a original guidance, or guidance per iter also.')
    args = parser.parse_args()

    app_dir = os.path.join(up(up(up(os.path.abspath(__file__)))), args.app_dir)
    framework = args.task[0]
    inference_task = args.task[1]

    label_config_path_framework_str_dicts = dict(deepedit='deepedit', deepeditplusplus='deepeditPlusPlus') 
    label_config_mapping_path = os.path.join(app_dir, "monailabel", label_config_path_framework_str_dicts[framework], args.label_configs)
    

    dataset_name = args.studies[:-9]
    dataset_subset = args.studies[-8:]

    
    if len(args.datetime) == 1 and len(args.checkpoint) == 1:
        datetime = args.datetime[0]
        checkpoint = args.checkpoint[0]

        if inference_task == "deepedit":
            initialisation = args.task[2]
            num_clicking_iters = args.task[3] 

            inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}_{initialisation}_initialisation_numIters_{num_clicking_iters}/{datetime}/{checkpoint}/run_{args.infer_run}'
            inference_image_subdirectory = 'datasets/' + inference_image_subtask
        else:
            inference_image_subtask = dataset_name + f'/{framework}/{dataset_subset}_{inference_task}/{datetime}/{checkpoint}/run_{args.infer_run}'
            inference_image_subdirectory = 'datasets/' + inference_image_subtask
    



    metric_score_configs_configs = [os.path.join(app_dir, args.metric_config[0], inference_image_subtask), args.metric_config[1], args.metric_config[2], args.metric_config[3]] 

    #If we are computing just for one image, to generate the png files (this is so that we can also separate out the guidances by iter for the deepedit original segmentations )
    if args.output_type == "png":
        save_folder = os.path.join(os.path.expanduser('~'), 'Debugging DeepEdit segmentations', metric_score_configs_configs[1] ,inference_image_subtask, 'png display')

        inference_images = [[os.path.join(app_dir, inference_image_subdirectory), image_name] for image_name in args.image_names]

        if inference_task == "deepedit":
            for inference_image in inference_images:
                output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework,  initialisation = plotting_func(inference_image, args.task, label_config_mapping_path, save_folder)

                plotting_images(output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework,  initialisation, args.output_type, metric_score_configs_configs)
        elif inference_task == "deepgrow":
            for inference_image in inference_images:
                output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework, initialisation = plotting_func(inference_image, args.task, label_config_mapping_path, save_folder)

                plotting_images(output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework, initialisation, args.output_type, metric_score_configs_configs)
    elif args.output_type == "tensorboard":

        save_folder = os.path.join(os.path.expanduser('~'), 'Debugging DeepEdit segmentations', metric_score_configs_configs[1],inference_image_subtask, 'tensorboard_display')

        inference_images = [[os.path.join(app_dir, inference_image_subdirectory), image_name] for image_name in os.listdir(os.path.join(app_dir, inference_image_subdirectory)) if image_name.endswith('.nii.gz')]

        for inference_image in inference_images:
            output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework,  initialisation = plotting_func(inference_image, args.task, label_config_mapping_path, save_folder)

            plotting_images(output_save_folder, seg_names, loading_transforms_output_dictionary, discrepancies_output_dictionary, guidance_points_plotting, guidances_per_iter_plotting, framework,  initialisation, args.output_type)
