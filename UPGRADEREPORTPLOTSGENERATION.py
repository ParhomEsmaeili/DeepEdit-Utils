import os
import csv
import numpy as np 
import argparse
# import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import wilcoxon

########################### NOTE TO SELF: When refactoring utilities code, we need to also think about how we are going to refactor the main bulk of the code.
############ WHY? Because the folders in which we save outputs, etc., need to be delineated by the version which is being tested. Therefore, we need to think about
####### how we want to refactor the bulk of the code to allow us to perform experimentation that is clearly delineated by deepedit++ versions perhaps? (i.e. v1.1.1, v1.1.2, etc)
####### and as such, then we can make use of the results that are also carefully delineated in this manner also.

################################## In order to generalise this piece of code also, we need to be capable of delineating between different score results also..

# KEEP MODULARISING, just more and more modularisation.

def wilcoxon_test(set_a, set_b):
    print(wilcoxon(set_a, set_b))
    print('\n')
    
def wilcoxon_test_multi(array_1, array_2):
    #we compare dice scores between two methods at each iteration

    #extracting initialisations:

    # array_1_initialisation = array_1[:,0]
    # array_2_initialisation = array_2[:,0]


    for i in [0,1,5,10]:
        print(f'iter {i} test \n')
        array_1_extract_col = array_1[:, i]
        array_2_extract_col = array_2[:, i]

        # #find the difference since initialisation
        # array_1_difference = array_1_extract_col - array_1_initialisation 
        # array_2_difference = array_2_extract_col - array_2_initialisation 

        wilcoxon_test(array_1_extract_col, array_2_extract_col)


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

def dice_score_reformat(dice_scores):
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
    failure_dice = 0
    # print(dice_scores_array.shape)
    failure_images = dict()
    row_removal_indices = []
    #We do it up till shape -2 since we do not want to include the stdev and mean in the row removal list.
    for index in range(dice_scores_array.shape[0]): #np.array(dice_scores):
        sub_array = dice_scores_array[index, :]
        #print(sub_array)
        if np.any(sub_array < failure_dice):
            
            # print(sub_array)
            # print(image_names[index])

            failure_images[image_names[index]] = sub_array.tolist()
            row_removal_indices.append(index)
        
        final_dice_scores_array = np.delete(dice_scores_array, row_removal_indices, axis=0)

        final_dice_scores_array_scores = final_dice_scores_array
        # print(final_dice_scores_array.shape)
    return [image_names, final_dice_scores_array_scores]

def repeating_run_averaging(dice_scores_list, repeat_bool=True):
    dice_score_names = dice_scores_list[0]
    dice_scores_array = dice_scores_list[1]

    if repeat_bool == False:
            pass
    else:  
        num_samples = int(dice_scores_array.shape[0]/3)
        assert num_samples == 250
        dice_scores_array = (dice_scores_array[0:num_samples, :] + dice_scores_array[num_samples:2*num_samples, :] + dice_scores_array[2*num_samples:, :])/3
        dice_score_names = dice_score_names[0:num_samples]
    
    return [dice_score_names, dice_scores_array]

def summary_extraction(array_dict):
    #assumes a dict which contains full names list which is N_samples long, and dice score arrays which are N_samples x N_iters: 
    output_dict = dict() 

    for dict_key in array_dict.keys():
        # dice_scores_names = array_dict[dict_key][0]
        dice_scores_array = array_dict[dict_key][1]
        #subdict which we will save into the full output dict
        output_dict_subdict = dict() 

        per_iter_minimum_array = np.min(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_minimum_array'] = per_iter_minimum_array
        
        per_iter_lq_array = np.quantile(dice_scores_array, 0.25, axis=0)
        output_dict_subdict['per_iter_lq_array'] = per_iter_lq_array
        
        per_iter_median_array = np.quantile(dice_scores_array, 0.5, axis=0)
        output_dict_subdict['per_iter_median_array'] = per_iter_median_array
        

        #ADD THE PER ITER CHANGE TOO. 
        # per_iter_median_change_array = np.diff(per_iter_median_array)
        # output_dict_subdict['per_iter_median_change_array'] = per_iter_median_change_array

        # #ADD THE mean across the iters
        # per_iter_median_change_array_mean = np.mean(per_iter_median_change_array)
        # output_dict_subdict['per_iter_median_change_array_mean'] = per_iter_median_change_array_mean

        per_iter_uq_array = np.quantile(dice_scores_array, 0.75, axis=0)
        output_dict_subdict['per_iter_uq_array'] = per_iter_uq_array
        
        per_iter_iqr_array = per_iter_uq_array - per_iter_lq_array 
        output_dict_subdict['per_iter_iqr_array'] = per_iter_iqr_array
        
        per_iter_maximum_array = np.max(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_maximum_array'] = per_iter_maximum_array

        per_iter_means_array = np.mean(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_means_array'] = per_iter_means_array

        #ADD THE PER ITER CHANGE TOO. 
        # per_iter_means_change_array = np.diff(per_iter_means_array)
        # output_dict_subdict['per_iter_means_change_array'] = per_iter_means_change_array


        # #ADD THE mean across the iters
        # per_iter_means_change_array_mean = np.mean(per_iter_means_change_array)
        # output_dict_subdict['per_iter_means_change_array_mean'] = per_iter_means_change_array_mean

        per_iter_stdevs_array = np.std(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_stdevs_array'] = per_iter_stdevs_array

        output_dict[dict_key] = output_dict_subdict

    return output_dict

def per_sample_summary_extraction(array_dict, image_name_string): 
    #assumes a dict which contains dice score arrays which are N_samples x N_iters. This is for scenarios where we have multiple runs per sample and want to 
    #obtain some summary just for a single image sample, across all the dataset partitions.

    output_dict = dict() 

    for dict_key in array_dict.keys():
        dice_scores_names = array_dict[dict_key][0]
        #find the indices of where that image sample is:
        image_sample_indices = np.where(dice_scores_names == image_name_string)[0]


        dice_scores_array = array_dict[dict_key][1][image_sample_indices, :]
        #subdict which we will save into the full output dict

        output_dict_subdict = dict()

        per_iter_minimum_array = np.min(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_minimum_array'] = per_iter_minimum_array
        
        per_iter_lq_array = np.quantile(dice_scores_array, 0.25, axis=0)
        output_dict_subdict['per_iter_lq_array'] = per_iter_lq_array
        
        per_iter_median_array = np.quantile(dice_scores_array, 0.5, axis=0)
        output_dict_subdict['per_iter_median_array'] = per_iter_median_array
        

        #ADD THE PER ITER CHANGE TOO. 
        per_iter_median_change_array = np.diff(per_iter_median_array)
        output_dict_subdict['per_iter_median_change_array'] = per_iter_median_change_array

        # #ADD THE mean across the iters
        # per_iter_median_change_array_mean = np.mean(per_iter_median_change_array)
        # output_dict_subdict['per_iter_median_change_array_mean'] = per_iter_median_change_array_mean

        per_iter_uq_array = np.quantile(dice_scores_array, 0.75, axis=0)
        output_dict_subdict['per_iter_uq_array'] = per_iter_uq_array
        
        per_iter_iqr_array = per_iter_uq_array - per_iter_lq_array 
        output_dict_subdict['per_iter_iqr_array'] = per_iter_iqr_array
        
        per_iter_maximum_array = np.max(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_maximum_array'] = per_iter_maximum_array

        per_iter_means_array = np.mean(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_means_array'] = per_iter_means_array

        #ADD THE PER ITER CHANGE TOO. 
        per_iter_means_change_array = np.diff(per_iter_means_array)
        output_dict_subdict['per_iter_means_change_array'] = per_iter_means_change_array


        # #ADD THE mean across the iters
        # per_iter_means_change_array_mean = np.mean(per_iter_means_change_array)
        # output_dict_subdict['per_iter_means_change_array_mean'] = per_iter_means_change_array_mean

        per_iter_stdevs_array = np.std(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_stdevs_array'] = per_iter_stdevs_array

        output_dict[dict_key] = output_dict_subdict
    
    
    return output_dict

def save_summarised(output_path, output_dict):
    #output dict contains a set of summary statistics which are separated by the fold configuration which they belong to
    #output path just denotes the sub-problem that they correspond to.
    
    os.makedirs(output_path, exist_ok=True)

    for folds_key in output_dict.keys():
        #save the snapshot name first
        dataset_snapshot_name = [folds_key]
        #adding a blank row


        with open(os.path.join(output_path, 'summary_across_snapshots.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(dataset_snapshot_name)

        snapshot_summary_dict = output_dict[folds_key]

        for summary_statistic_key in snapshot_summary_dict.keys():
            summary_stat_row = [summary_statistic_key]
            for score in snapshot_summary_dict[summary_statistic_key]:
                summary_stat_row.append(score) 

            with open(os.path.join(output_path, 'summary_across_snapshots.csv'),'a') as f:
                
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(summary_stat_row)            


    return 

def save_sample_summarised(output_path, sample, output_dict):
    #output dict contains a set of summary statistics which are separated by the fold configuration which they belong to
    #output path just denotes the sub-problem that they correspond to.
    
    os.makedirs(output_path, exist_ok=True)

    for folds_key in output_dict.keys():
        #save the snapshot name first
        dataset_snapshot_name = [folds_key]
        #adding a blank row


        with open(os.path.join(output_path, f'summary_across_snapshots_{sample[:-7]}.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(dataset_snapshot_name)

        snapshot_summary_dict = output_dict[folds_key]

        for summary_statistic_key in snapshot_summary_dict.keys():
            summary_stat_row = [summary_statistic_key]
            for score in snapshot_summary_dict[summary_statistic_key]:
                summary_stat_row.append(score) 

            with open(os.path.join(output_path, f'summary_across_snapshots_{sample[:-7]}.csv'),'a') as f:
                
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(summary_stat_row)            


    return 

def setBoxColors(bp, edge_colour, fill_colour):
    for sub_type_index in range(len(edge_colour)):
        edge_colour_specific = edge_colour[sub_type_index]
        fill_colour_specific = fill_colour[sub_type_index]

        for element in ['boxes', 'medians']:
            plt.setp(bp[element][sub_type_index], color=edge_colour_specific)
        
        for element in ['whiskers', 'caps']:
            plt.setp(bp[element][2 * sub_type_index], color=edge_colour_specific)
            plt.setp(bp[element][2 * sub_type_index + 1], color=edge_colour_specific)

        if sub_type_index <= 1:
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific)
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific)
        elif sub_type_index > 1 and sub_type_index <= 3:
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific) #, hatch=r"//")
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific) #, hatch=r"//")
        elif sub_type_index == 4:
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific)
            plt.setp(bp['boxes'][sub_type_index], facecolor=fill_colour_specific)
            # patch.set(facecolor=fill_colour_specific)

def plot_boxplots(output_path, nnunet_list, deepedit_autoseg_init_list, deepedit_deepgrow_init_list, deepeditpp_autoseg_init_list, deepeditpp_deepgrow_init_list):

    # nnunet_summary = nnunet_list[0]
    # deepedit_autoseg_init_summary = deepedit_autoseg_init_list[0]
    # deepedit_deepgrow_init_summary = deepedit_deepgrow_init_list[0]
    # deepeditplusplus_autoseg_init_summary = deepeditpp_autoseg_init_list[0]
    # deepeditplusplus_deepgrow_init_summary = deepeditpp_deepgrow_init_list[0]

    percentages = ['25%', '50%', '75%', '100%']
    nested_scores_autoseg_init_dict = dict() #Within each percentage, we order them as DeepEdit init, DeepEdit++ init, DEepEdit + 10 click, DeepEdit++ + 10 click
    #if percentage = 100% then also append the nnunet score.
    nested_scores_deepgrow_init_dict = dict() 

    nested_scores_autoseg_init_dict_deepeditpp = dict() #Within each percentage we only have deepedit++ init and deepedit++ + 10 clicks
    nested_scores_deepgrow_init_dict_deepeditpp = dict()

    for i in percentages:
        #For each configuration, we append a list with each set of scores outlined above.
        
        nested_scores_autoseg_init_list = list()

        nested_scores_autoseg_init_list.append(deepedit_autoseg_init_list[1][i][1][:,0])
        nested_scores_autoseg_init_list.append(deepeditpp_autoseg_init_list[1][i][1][:,0])
        nested_scores_autoseg_init_list.append(deepedit_autoseg_init_list[1][i][1][:,-1])
        nested_scores_autoseg_init_list.append(deepeditpp_autoseg_init_list[1][i][1][:,-1])

        if i == "100%":
            nested_scores_autoseg_init_list.append(nnunet_list[1][i][1][:,0])
        
        nested_scores_autoseg_init_dict[i] = nested_scores_autoseg_init_list
    
    for i in percentages:
        #For each configuration, we append a list with each set of scores outlined above.
        
        nested_scores_deepgrow_init_list = list()

        nested_scores_deepgrow_init_list.append(deepedit_deepgrow_init_list[1][i][1][:,0])
        nested_scores_deepgrow_init_list.append(deepeditpp_deepgrow_init_list[1][i][1][:,0])
        nested_scores_deepgrow_init_list.append(deepedit_deepgrow_init_list[1][i][1][:,-1])
        nested_scores_deepgrow_init_list.append(deepeditpp_deepgrow_init_list[1][i][1][:,-1])

        if i == "100%":
            nested_scores_deepgrow_init_list.append(nnunet_list[1][i][1][:,0])
        
        nested_scores_deepgrow_init_dict[i] = nested_scores_deepgrow_init_list


    for i in percentages:
        #For each deepeditpp configuration, we append a list with each set of scores outlined above.
        
        nested_scores_autoseg_init_list = list()

        # nested_scores_autoseg_init_list.append(deepedit_autoseg_init_list[1][i][1][:,0])
        nested_scores_autoseg_init_list.append(deepeditpp_autoseg_init_list[1][i][1][:,0])
        # nested_scores_autoseg_init_list.append(deepedit_autoseg_init_list[1][i][1][:,-1])
        nested_scores_autoseg_init_list.append(deepeditpp_autoseg_init_list[1][i][1][:,-1])

        if i == "100%":
            nested_scores_autoseg_init_list.append(nnunet_list[1][i][1][:,0])
        
        nested_scores_autoseg_init_dict_deepeditpp[i] = nested_scores_autoseg_init_list
    
    for i in percentages:
        #For each deepeditpp, we append a list with each set of scores outlined above.
        
        nested_scores_deepgrow_init_list = list()

        # nested_scores_deepgrow_init_list.append(deepedit_deepgrow_init_list[1][i][1][:,0])
        nested_scores_deepgrow_init_list.append(deepeditpp_deepgrow_init_list[1][i][1][:,0])
        # nested_scores_deepgrow_init_list.append(deepedit_deepgrow_init_list[1][i][1][:,-1])
        nested_scores_deepgrow_init_list.append(deepeditpp_deepgrow_init_list[1][i][1][:,-1])

        if i == "100%":
            nested_scores_deepgrow_init_list.append(nnunet_list[1][i][1][:,0])
        
        nested_scores_deepgrow_init_dict_deepeditpp[i] = nested_scores_deepgrow_init_list

    # fold_1 = nested_scores_autoseg_init_dict['25%']
    # fold_1_2 = nested_scores_autoseg_init_dict['50%']
    # fold_1_2_3 = nested_scores_autoseg_init_dict['75%']
    # fold_1_2_3_4 = nested_scores_autoseg_init_dict['100%']

    # plt.figure() 
    # bpa = plt.boxplot(fold_1, positions = [1, 2, 3, 4], widths = 0.6)
    # bpa = plt.boxplot(fold_1_2, positions = [6, 7, 8, 9], widths = 0.6)
    # bpa = plt.boxplot(fold_1_2_3, positions = [11, 12, 13, 14], widths = 0.6)
    # bpa = plt.boxplot(fold_1_2_3_4, positions = [16, 17, 18, 19, 20], widths = 0.6)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_autoseg_init.png'))
    # plt.close()

    # plt.figure()
    # fold_1 = nested_scores_deepgrow_init_dict['25%']
    # fold_1_2 = nested_scores_deepgrow_init_dict['50%']
    # fold_1_2_3 = nested_scores_deepgrow_init_dict['75%']
    # fold_1_2_3_4 = nested_scores_deepgrow_init_dict['100%']

    # bpb = plt.boxplot(fold_1, positions = [1, 2, 3, 4], widths = 0.6)
    # bpb = plt.boxplot(fold_1_2, positions = [6, 7, 8, 9], widths = 0.6)
    # bpb = plt.boxplot(fold_1_2_3, positions = [11, 12, 13, 14], widths = 0.6)
    # bpb = plt.boxplot(fold_1_2_3_4, positions = [16, 17, 18, 19, 20], widths = 0.6)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_deepgrow_init.png'))
    # plt.close()

    ################################################################################################
    # plt.figure()
    # fold_1 = nested_scores_autoseg_init_dict_deepeditpp['25%']
    # fold_1_2 = nested_scores_autoseg_init_dict_deepeditpp['50%']
    # fold_1_2_3 = nested_scores_autoseg_init_dict_deepeditpp['75%']
    # fold_1_2_3_4 = nested_scores_autoseg_init_dict_deepeditpp['100%']

    # bpc = plt.boxplot(fold_1, positions = [1, 2], widths = 0.6)
    # bpc = plt.boxplot(fold_1_2, positions = [4,5], widths = 0.6)
    # bpc = plt.boxplot(fold_1_2_3, positions = [7,8], widths = 0.6)
    # bpc = plt.boxplot(fold_1_2_3_4, positions = [10, 11, 13], widths = 0.6)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_autoseg_init_deepeditpp.png'))
    # plt.close()

    
    # plt.figure()
    # fold_1 = nested_scores_deepgrow_init_dict_deepeditpp['25%']
    # fold_1_2 = nested_scores_deepgrow_init_dict_deepeditpp['50%']
    # fold_1_2_3 = nested_scores_deepgrow_init_dict_deepeditpp['75%']
    # fold_1_2_3_4 = nested_scores_deepgrow_init_dict_deepeditpp['100%']

    # bpd = plt.boxplot(fold_1, positions = [1, 2], widths = 0.6)
    # bpd = plt.boxplot(fold_1_2, positions = [4,5], widths = 0.6)
    # bpd = plt.boxplot(fold_1_2_3, positions = [7,8], widths = 0.6)
    # bpd = plt.boxplot(fold_1_2_3_4, positions = [10, 11, 13], widths = 0.6)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_deepgrow_init_deepeditpp.png'))
    # plt.close()


    ########################################################################################################################
    #modifying the fliers

    fold_1 = nested_scores_autoseg_init_dict['25%']
    fold_1_2 = nested_scores_autoseg_init_dict['50%']
    fold_1_2_3 = nested_scores_autoseg_init_dict['75%']
    fold_1_2_3_4 = nested_scores_autoseg_init_dict['100%']

    #increasing figsize maintaining aspect ratio
    fig = plt.figure()
    # fig.set_rasterized(True)
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)


    ax = plt.axes()
    bpe = plt.boxplot(fold_1, positions = [1, 2, 3, 4], widths = 0.6, showfliers=False, patch_artist=True)

    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])
    
    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1[i] < whiskers_lower[i], fold_1[i] > whiskers_upper[i]), fold_1[i]) for i in range(4)]
    # outliers = [np.extract(np.logical_or(0 <= fold_1[i], 1 >= fold_1[i]), fold_1[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.04, size=len(outliers[j])) for j, i in enumerate([1,2,3,4])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    bpe = ax.boxplot(fold_1_2, positions = [6, 7, 8, 9], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1_2[i] < whiskers_lower[i], fold_1_2[i] > whiskers_upper[i]), fold_1_2[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([6,7,8,9])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    bpe = ax.boxplot(fold_1_2_3, positions = [11, 12, 13, 14], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1_2_3[i] < whiskers_lower[i], fold_1_2_3[i] > whiskers_upper[i]), fold_1_2_3[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([11,12,13,14])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)


    bpe = ax.boxplot(fold_1_2_3_4, positions = [16, 17, 18, 19, 20], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple','orange'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen', 'orange'])
    
    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(5)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(5)]
    outliers = [np.extract(np.logical_or(fold_1_2_3_4[i] < whiskers_lower[i], fold_1_2_3_4[i] > whiskers_upper[i]), fold_1_2_3_4[i]) for i in range(5)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([16, 17, 18, 19, 20])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    ax.set_xticks([2.5, 7.5, 12.5, 18])
    ax.set_xticklabels(['25%', '50%', '75%', '100%'])
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xlabel('Data Partition Configuration', fontsize=14, labelpad=20)
    ax.set_ylabel('Dice Score', fontsize=14, labelpad=15)
    ax.set_title('Comparison across DeepEdit and DeepEdit++ Inference Runs with Automatic Initialisation, and a Fully Automatic nnU-Net Baseline', fontsize=14, pad=25)
    
    skyblue_patch = mpatches.Patch(color='skyblue', label='D.E Init.')
    violet_patch = mpatches.Patch(color='violet', label='D.E ++ Init.')
    aquamarine_patch = mpatches.Patch(color='aquamarine', label='D.E & 10 Edit Iters')
    yellowgreen_patch = mpatches.Patch(color='yellowgreen', label='D.E ++ & 10 Edit Iters')
    orange_patch = mpatches.Patch(color='orange', label='nnU-Net')
    # ['palegreen', 'violet', 'aquamarine', 'mediumpurple','orange']
    ax.legend(handles=[skyblue_patch, violet_patch, aquamarine_patch, yellowgreen_patch, orange_patch], bbox_to_anchor=(1.02, 0.5), loc="center left")
    # l5 = plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
    #             bbox_transform=fig.transFigure, ncol=3)
    os.makedirs(output_path, exist_ok=True)
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(output_path, 'testing_autoseg_init_modified.pdf'), bbox_inches='tight') #, rasterized=True)
    plt.close()

    ######################################################

   
    ################################################################################################
    # plt.figure()
    # fold_1 = nested_scores_autoseg_init_dict_deepeditpp['25%']
    # fold_1_2 = nested_scores_autoseg_init_dict_deepeditpp['50%']
    # fold_1_2_3 = nested_scores_autoseg_init_dict_deepeditpp['75%']
    # fold_1_2_3_4 = nested_scores_autoseg_init_dict_deepeditpp['100%']

    # bpg = plt.boxplot(fold_1, positions = [1, 2], widths = 0.6, showfliers=False)
    # bpg = plt.boxplot(fold_1_2, positions = [4,5], widths = 0.6, showfliers=False)
    # bpg = plt.boxplot(fold_1_2_3, positions = [7,8], widths = 0.6, showfliers=False)
    # bpg = plt.boxplot(fold_1_2_3_4, positions = [10, 11, 13], widths = 0.6, showfliers=False)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_autoseg_init_deepeditpponly_modified.png'))
    # plt.close()

    
    # plt.figure()
    # fold_1 = nested_scores_deepgrow_init_dict_deepeditpp['25%']
    # fold_1_2 = nested_scores_deepgrow_init_dict_deepeditpp['50%']
    # fold_1_2_3 = nested_scores_deepgrow_init_dict_deepeditpp['75%']
    # fold_1_2_3_4 = nested_scores_deepgrow_init_dict_deepeditpp['100%']

    # bph = plt.boxplot(fold_1, positions = [1, 2], widths = 0.6, showfliers=False)
    # bph = plt.boxplot(fold_1_2, positions = [4,5], widths = 0.6, showfliers=False)
    # bph = plt.boxplot(fold_1_2_3, positions = [7,8], widths = 0.6,showfliers=False)
    # bph = plt.boxplot(fold_1_2_3_4, positions = [10, 11, 13], widths = 0.6, showfliers=False)

    # os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'testing_deepgrow_init_deepeditpponly_modified.png'))
    # plt.close()

    fold_1 = nested_scores_deepgrow_init_dict['25%']
    fold_1_2 = nested_scores_deepgrow_init_dict['50%']
    fold_1_2_3 = nested_scores_deepgrow_init_dict['75%']
    fold_1_2_3_4 = nested_scores_deepgrow_init_dict['100%']

    #increasing figsize maintaining aspect ratio
    fig = plt.figure()
    # fig.set_rasterized(True)
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)


    ax = plt.axes()
    bpe = plt.boxplot(fold_1, positions = [1, 2, 3, 4], widths = 0.6, showfliers=False, patch_artist=True)

    # setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1[i] < whiskers_lower[i], fold_1[i] > whiskers_upper[i]), fold_1[i]) for i in range(4)]
    # outliers = [np.extract(np.logical_or(0 <= fold_1[i], 1 >= fold_1[i]), fold_1[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.04, size=len(outliers[j])) for j, i in enumerate([1,2,3,4])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    bpe = ax.boxplot(fold_1_2, positions = [6, 7, 8, 9], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1_2[i] < whiskers_lower[i], fold_1_2[i] > whiskers_upper[i]), fold_1_2[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([6,7,8,9])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    bpe = ax.boxplot(fold_1_2_3, positions = [11, 12, 13, 14], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(4)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(4)]
    outliers = [np.extract(np.logical_or(fold_1_2_3[i] < whiskers_lower[i], fold_1_2_3[i] > whiskers_upper[i]), fold_1_2_3[i]) for i in range(4)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([11,12,13,14])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)


    bpe = ax.boxplot(fold_1_2_3_4, positions = [16, 17, 18, 19, 20], widths = 0.6, showfliers=False, patch_artist=True)
    # setBoxColors(bpe, ['black', 'black', 'black', 'black', 'black'], ['palegreen', 'violet', 'aquamarine', 'mediumpurple','orange'])
    setBoxColors(bpe, ['black', 'black', 'black', 'black','black'], ['skyblue', 'violet', 'aquamarine', 'yellowgreen','orange'])

    whiskers_lower = [(bpe['whiskers'][2 * i].get_data())[1][1] for i in range(5)]
    whiskers_upper = [(bpe['whiskers'][2 * i + 1].get_data())[1][1] for i in range(5)]
    outliers = [np.extract(np.logical_or(fold_1_2_3_4[i] < whiskers_lower[i], fold_1_2_3_4[i] > whiskers_upper[i]), fold_1_2_3_4[i]) for i in range(5)]
    x_axis = [np.random.normal(i, 0.02, size=len(outliers[j])) for j, i in enumerate([16, 17, 18, 19, 20])]

    flattened_outliers = []
    for sub_array in outliers:
        for value in sub_array:
            flattened_outliers.append(value)
    flattened_x_axis = [] 
    for sub_array in x_axis:
        for value in sub_array:
            flattened_x_axis.append(value)

    ax.plot(flattened_x_axis, flattened_outliers, 'r.', alpha=0.35)

    ax.set_xticks([2.5, 7.5, 12.5, 18])
    ax.set_xticklabels(['25%', '50%', '75%', '100%'])
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xlabel('Data Partition Configuration', fontsize=14, labelpad=20)
    ax.set_ylabel('Dice Score', fontsize=14, labelpad=15)
    ax.set_title('Comparison across DeepEdit and DeepEdit++ Inference Runs with Interactive Initialisation, and a Fully Automatic nnU-Net Baseline', fontsize=14, pad=25)
    
    # palegreen_patch = mpatches.Patch(color='palegreen', label='D.E Init.')
    # violet_patch = mpatches.Patch(color='violet', label='D.E ++ Init.')
    # aquamarine_patch = mpatches.Patch(color='aquamarine', label='D.E & 10 Edit Iters')
    # mediumpurple_patch = mpatches.Patch(color='mediumpurple', label='D.E ++ & 10 Edit Iters')
    # orange_patch = mpatches.Patch(color='orange', label='nnU-Net')
    # # ['palegreen', 'violet', 'aquamarine', 'mediumpurple','orange']
    # ax.legend(handles=[palegreen_patch, violet_patch, aquamarine_patch, mediumpurple_patch, orange_patch], bbox_to_anchor=(1.02, 0.5), loc="center left")

    skyblue_patch = mpatches.Patch(color='skyblue', label='D.E Init.')
    violet_patch = mpatches.Patch(color='violet', label='D.E ++ Init.')
    aquamarine_patch = mpatches.Patch(color='aquamarine', label='D.E & 10 Edit Iters')
    yellowgreen_patch = mpatches.Patch(color='yellowgreen', label='D.E ++ & 10 Edit Iters')
    orange_patch = mpatches.Patch(color='orange', label='nnU-Net')
    # ['palegreen', 'violet', 'aquamarine', 'mediumpurple','orange']
    ax.legend(handles=[skyblue_patch, violet_patch, aquamarine_patch, yellowgreen_patch, orange_patch], bbox_to_anchor=(1.02, 0.5), loc="center left")

    # l5 = plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
    #             bbox_transform=fig.transFigure, ncol=3)
    os.makedirs(output_path, exist_ok=True)
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(output_path, 'testing_deepgrow_init_modified.pdf'), bbox_inches='tight') #, rasterized=True)
    plt.close()

    return 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='BraTS2021_01620.nii.gz')
    args = parser.parse_args()


    root_folder = '/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/results/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT'
    image_name = args.image_name #"BraTS2021_01620.nii.gz" 
    epoch = 'best_val_score_epoch'

    deepediting_infer_runs = 'run_0_1_2'

    nnunet_infer_runs = 'run_0'

    root_folder_deepedit = os.path.join(root_folder,'deepedit')
    root_folder_deepeditplusplus = os.path.join(root_folder, 'deepeditplusplus')
    root_folder_nnunet = os.path.join(root_folder, 'nnUnet')

   
    deepedit_dataset_versions = ['19072024_183616', '20072024_141015', '20072024_174858', '20072024_210351'] #need to populate this with the actual ones for out of the box performance.
    deepeditplusplus_dataset_versions = ['10072024_182402', '12072024_164436', '18072024_004553', '15072024_010034'] #New version is going to be this instead ['10072024_182402', '12072024_164436', '18072024_004553', '15072024_010034'] 

    #when nnunet versions are done training then change this back. 

    nnunet_dataset_versions = ['nnUnet_fold_1_2_3_4']#['nnUnet_fold_1_2_3_4', 'nnUnet_fold_1_2_3', 'nnUnet_fold_1_2', 'nnUnet_fold_1']


    ### string which encodes what the task is:

    autoseg_only_runtask = 'imagesTs_autoseg'

    autoseg_init_runtask = 'imagesTs_deepedit_autoseg_initialisation_numIters_10'


    interactive_init_runtask = 'imagesTs_deepedit_deepgrow_initialisation_numIters_10'

    #we are going to save in order from largest to smallest dataset partition

    deepedit_data_stores_per_version_autoseg = dict()

    deepedit_data_stores_per_version_deepgrow = dict()



    deepeditplusplus_data_stores_per_version_autoseg = dict()

    deepeditplusplus_data_stores_per_version_deepgrow = dict()


    nnunet_data_stores_per_version_autoseg = dict()


    folds_list_names = ['100%', '75%', '50%', '25%']


    #we conjoin the fold list names with the dataset versions so we send the file to the corresponding spot.
    
    #deepedit first

    fold_model_version_datetime_conjoined = dict()

    for index, model_version in enumerate(deepedit_dataset_versions):

        fold_percentage = folds_list_names[index]
        fold_model_version_datetime_conjoined[fold_percentage] = model_version


    #deepeditplusplus
    for index, model_version in enumerate(deepeditplusplus_dataset_versions):

        fold_percentage = folds_list_names[index]
        fold_model_version_datetime_conjoined[fold_percentage] = model_version


    #nnunet 
    for index, model_version in enumerate(nnunet_dataset_versions):

        fold_percentage = folds_list_names[index]
        fold_model_version_datetime_conjoined[fold_percentage] = model_version



    

    #extract the scores 
    for index, version in enumerate(deepedit_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepedit, autoseg_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        results_path_deepgrow_init = os.path.join(root_folder_deepedit, interactive_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # print(folds_list_names[index])
        # print(results_path_autoseg_init)
        # print(results_path_deepgrow_init)
        deepedit_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=True)
        deepedit_data_stores_per_version_deepgrow[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_deepgrow_init)), repeat_bool=True)


    for index,version in enumerate(deepeditplusplus_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepeditplusplus, autoseg_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        results_path_deepgrow_init = os.path.join(root_folder_deepeditplusplus, interactive_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # print(folds_list_names[index])
        # print(results_path_autoseg_init)
        # print(results_path_deepgrow_init)
        deepeditplusplus_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=True)
        deepeditplusplus_data_stores_per_version_deepgrow[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_deepgrow_init)), repeat_bool=True)

    for index,version in enumerate(nnunet_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_nnunet, autoseg_only_runtask, version, epoch, 'run_collection', nnunet_infer_runs)
        # results_path_deepgrow_init = os.path.join(root_folder_nnunet, interactive_init_runtask, version, epoch)
        # print(folds_list_names[index])
        # print(results_path_autoseg_init)
        nnunet_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=False)
        # deepeditplusplus_data_stores_per_version_deepgrow.append(dice_score_extraction(results_path_autoseg_init))


    #the data is saved as an N_samples x N_iters array, here we extract the scores which we will use going forward? we are going to extract means, medians, iqr, min and max.

    summary_scores_deepedit_autoseg_init = summary_extraction(deepedit_data_stores_per_version_autoseg)
    summary_scores_deepedit_deepgrow_init = summary_extraction(deepedit_data_stores_per_version_deepgrow)

    summary_scores_deepeditplusplus_autoseg_init = summary_extraction(deepeditplusplus_data_stores_per_version_autoseg)
    summary_scores_deepeditplusplus_deepgrow_init = summary_extraction(deepeditplusplus_data_stores_per_version_deepgrow)

    summary_scores_nnunet_autoseg_only =  summary_extraction(nnunet_data_stores_per_version_autoseg) 


    ######################################################################################################################

    #computing summary statistics for a given sample

    # summary_scores_single_sample_deepedit_autoseg_init = per_sample_summary_extraction(deepedit_data_stores_per_version_autoseg, image_name)
    # summary_scores_single_sample_deepedit_deepgrow_init = per_sample_summary_extraction(deepedit_data_stores_per_version_deepgrow, image_name)

    # summary_scores_single_sample_deepeditplusplus_autoseg_init = per_sample_summary_extraction(deepeditplusplus_data_stores_per_version_autoseg, image_name)
    # summary_scores_single_sample_deepeditplusplus_deepgrow_init = per_sample_summary_extraction(deepeditplusplus_data_stores_per_version_deepgrow, image_name)

    # summary_scores_single_sample_nnunet_autoseg_only =  per_sample_summary_extraction(nnunet_data_stores_per_version_autoseg, image_name) 



    # saving the outputs to a csv file which contains all the summaries across all folders....

    #model versions conjoined 

    nnunet_model_versions_conjoined = '_'.join(nnunet_dataset_versions)
    deepedit_model_versions_conjoined = '_'.join(deepedit_dataset_versions)
    deepeditplusplus_model_versions_conjoined = '_'.join(deepeditplusplus_dataset_versions)




    root_summaries_folder = '/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/results_summaries/results_spreadsheet_summaries/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT'
    # image_name = args.image_name #"BraTS2021_01620.nii.gz" 
    epoch = 'best_val_score_epoch'

    deepediting_infer_runs = 'run_0_1_2'

    nnunet_infer_runs = 'run_0'

    summary_root_folder_deepedit = os.path.join(root_summaries_folder,'deepedit')
    summary_root_folder_deepeditplusplus = os.path.join(root_summaries_folder, 'deepeditplusplus')
    summary_root_folder_nnunet = os.path.join(root_summaries_folder, 'nnUnet')


    autoseg_only_runtask = 'imagesTs_autoseg'

    autoseg_init_runtask = 'imagesTs_deepedit_autoseg_initialisation_numIters_10'


    interactive_init_runtask = 'imagesTs_deepedit_deepgrow_initialisation_numIters_10'

    ### aggregating by the inference task runs
    task_root_folder_deepedit_autoseg_init = os.path.join(summary_root_folder_deepedit, autoseg_init_runtask)
    task_root_folder_deepedit_deepgrow_init = os.path.join(summary_root_folder_deepedit, interactive_init_runtask)

    task_root_folder_deepeditplusplus_autoseg_init = os.path.join(summary_root_folder_deepeditplusplus, autoseg_init_runtask)
    task_root_folder_deepeditplusplus_deepgrow_init = os.path.join(summary_root_folder_deepeditplusplus, interactive_init_runtask)

    task_root_folder_nnunet_autoseg = os.path.join(summary_root_folder_nnunet, autoseg_only_runtask)


    #aggregating by the model versions (across the dataset versions) for all of those inference task runs
    model_versions_aggregate_save_root_path_nnunet_autoseg = os.path.join(task_root_folder_nnunet_autoseg, nnunet_model_versions_conjoined)

    model_versions_aggregate_save_root_path_deepedit_autoseg_init = os.path.join(task_root_folder_deepedit_autoseg_init, deepedit_model_versions_conjoined)
    model_versions_aggregate_save_root_path_deepedit_deepgrow_init = os.path.join(task_root_folder_deepedit_deepgrow_init, deepedit_model_versions_conjoined)

    model_versions_aggregate_save_root_path_deepeditplusplus_autoseg_init = os.path.join(task_root_folder_deepeditplusplus_autoseg_init, deepeditplusplus_model_versions_conjoined)
    model_versions_aggregate_save_root_path_deepeditplusplus_deepgrow_init = os.path.join(task_root_folder_deepeditplusplus_deepgrow_init, deepeditplusplus_model_versions_conjoined)


    #pointing path to epoch/infer run collection
    epoch_path_nnunet = os.path.join(model_versions_aggregate_save_root_path_nnunet_autoseg, epoch)

    run_collection_path_nnunet = os.path.join(epoch_path_nnunet, 'run_collection', nnunet_infer_runs)


    #pointing path to epoch/infer run collection
    epoch_path_deepedit_autoseg_init = os.path.join(model_versions_aggregate_save_root_path_deepedit_autoseg_init, epoch)
    epoch_path_deepedit_deepgrow_init = os.path.join(model_versions_aggregate_save_root_path_deepedit_deepgrow_init, epoch)


    run_collection_path_deepedit_autoseg_init = os.path.join(epoch_path_deepedit_autoseg_init, 'run_collection', deepediting_infer_runs)
    run_collection_path_deepedit_deepgrow_init = os.path.join(epoch_path_deepedit_deepgrow_init, 'run_collection', deepediting_infer_runs)


    #pointing path to epoch/infer run collection
    epoch_path_deepeditplusplus_autoseg_init = os.path.join(model_versions_aggregate_save_root_path_deepeditplusplus_autoseg_init, epoch)
    epoch_path_deepeditplusplus_deepgrow_init = os.path.join(model_versions_aggregate_save_root_path_deepeditplusplus_deepgrow_init, epoch)

    run_collection_path_deepeditplusplus_autoseg_init = os.path.join(epoch_path_deepeditplusplus_autoseg_init, 'run_collection', deepediting_infer_runs)
    run_collection_path_deepeditplusplus_deepgrow_init = os.path.join(epoch_path_deepeditplusplus_deepgrow_init, 'run_collection', deepediting_infer_runs)

    print(run_collection_path_nnunet)
    print(run_collection_path_deepedit_autoseg_init)
    print(run_collection_path_deepedit_deepgrow_init)
    print(run_collection_path_deepeditplusplus_autoseg_init)
    print(run_collection_path_deepeditplusplus_deepgrow_init)

    save_summarised(run_collection_path_nnunet, summary_scores_nnunet_autoseg_only)
    save_summarised(run_collection_path_deepedit_autoseg_init, summary_scores_deepedit_autoseg_init)
    save_summarised(run_collection_path_deepedit_deepgrow_init, summary_scores_deepedit_deepgrow_init)
    save_summarised(run_collection_path_deepeditplusplus_autoseg_init, summary_scores_deepeditplusplus_autoseg_init)
    save_summarised(run_collection_path_deepeditplusplus_deepgrow_init, summary_scores_deepeditplusplus_deepgrow_init)


    root_summaries_folder = '/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/results_summaries/results_plots/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT'

    plot_boxplots(root_summaries_folder, [summary_scores_nnunet_autoseg_only, nnunet_data_stores_per_version_autoseg], [summary_scores_deepedit_autoseg_init, deepedit_data_stores_per_version_autoseg], [summary_scores_deepedit_deepgrow_init, deepedit_data_stores_per_version_deepgrow], [summary_scores_deepeditplusplus_autoseg_init, deepeditplusplus_data_stores_per_version_autoseg], [summary_scores_deepeditplusplus_deepgrow_init, deepeditplusplus_data_stores_per_version_deepgrow])
        

    ##############################################

    #running statistical tests:

    #comparing autoseg init and interactive init within each framework and partition:

    #deepedit first:

    print('statistical test to compare autoseg and interactive seg initialisations for deepedit \n')

    for percentage in deepedit_data_stores_per_version_autoseg.keys():
        print(f'data partition configuration {percentage} \n')
        wilcoxon_test_multi(deepedit_data_stores_per_version_autoseg[percentage][1], deepedit_data_stores_per_version_deepgrow[percentage][1])

    #deepeditplusplus second:

    print('statistical test to compare autoseg and interactive seg initialisations for deepeditplusplus \n')
    
    for percentage in deepeditplusplus_data_stores_per_version_autoseg.keys():
        print(f'data partition configuration {percentage} \n')
        wilcoxon_test_multi(deepeditplusplus_data_stores_per_version_autoseg[percentage][1], deepeditplusplus_data_stores_per_version_deepgrow[percentage][1])


    #comparing autoseg init and interactive init between frameworks for each partition:


    print('statistical test to compare autoseg initialisations between deepedit and deepeditplusplus \n')
    
    for percentage in deepedit_data_stores_per_version_autoseg.keys():
        print(f'data partition configuration {percentage} \n')
        wilcoxon_test_multi(deepedit_data_stores_per_version_autoseg[percentage][1], deepeditplusplus_data_stores_per_version_autoseg[percentage][1])

    print('statistical test to compare interactive seg initialisations between deepedit and deepeditplusplus \n')
    
    for percentage in deepedit_data_stores_per_version_deepgrow.keys():
        print(f'data partition configuration {percentage} \n')
        wilcoxon_test_multi(deepedit_data_stores_per_version_deepgrow[percentage][1], deepeditplusplus_data_stores_per_version_deepgrow[percentage][1])


    print('statistical test to compare across dataset partitions for deepeditplusplus (not deepedit because they seem stochastic and not converged?) \n')

    print('we will be comparing first across autoseg initialisations \n')

    percentages_list = list(deepedit_data_stores_per_version_autoseg.keys())

    for i in range(1,len(percentages_list)):
        print(f'comparison between percentage {percentages_list[i]} and {percentages_list[i-1]} \n')

        wilcoxon_test_multi(deepeditplusplus_data_stores_per_version_autoseg[percentages_list[i]][1], deepeditplusplus_data_stores_per_version_autoseg[percentages_list[i-1]][1])
        
    print('we will be comparing first across deepgrow initialisations \n')

    percentages_list = list(deepedit_data_stores_per_version_deepgrow.keys())

    for i in range(1,len(percentages_list)):
        print(f'comparison between percentage {percentages_list[i]} and {percentages_list[i-1]} \n')

        wilcoxon_test_multi(deepeditplusplus_data_stores_per_version_deepgrow[percentages_list[i]][1], deepeditplusplus_data_stores_per_version_deepgrow[percentages_list[i-1]][1])
        
        

    print('we will be comparing nnU-net to the best deepeditplusplus outputs at 100% (iter 10 of automatic and interactive init) \n')

    
    print('comparison between nnu-net and auto init deepeditplusplus \n')

    wilcoxon_test(deepeditplusplus_data_stores_per_version_autoseg['100%'][1][:,-1], nnunet_data_stores_per_version_autoseg['100%'][1].flatten())

    print('comparison between nnu-net and interactive init deepeditplusplus \n')

    wilcoxon_test(deepeditplusplus_data_stores_per_version_deepgrow['100%'][1][:,-1], nnunet_data_stores_per_version_autoseg['100%'][1].flatten())


    print('comparison across iterations for deepedit with autoseg and interactive init')

    print('automatic init \n')
    for percentage in deepedit_data_stores_per_version_autoseg.keys():
        print(f'data partition configuration {percentage} \n')

        iters = [1,5,10]

        for iter in iters:
            print(f'comparing iter {iter} to init \n')
            wilcoxon_test(deepedit_data_stores_per_version_autoseg[percentage][1][:,iter], deepedit_data_stores_per_version_autoseg[percentage][1][:,0])

    print('interactive init \n')
    for percentage in deepedit_data_stores_per_version_deepgrow.keys():
        print(f'data partition configuration {percentage} \n')

        iters = [1,5,10]

        for iter in iters:
            print(f'comparing iter {iter} to init \n')
            wilcoxon_test(deepedit_data_stores_per_version_deepgrow[percentage][1][:,iter], deepedit_data_stores_per_version_deepgrow[percentage][1][:,0])
    

    print('comparison across iterations for deepeditplusplus with autoseg and interactive init')


    print('automatic init \n')
    for percentage in deepeditplusplus_data_stores_per_version_autoseg.keys():
        print(f'data partition configuration {percentage} \n')

        iters = [1,5,10]

        for iter in iters:
            print(f'comparing iter {iter} to init \n')
            wilcoxon_test(deepeditplusplus_data_stores_per_version_autoseg[percentage][1][:,iter], deepeditplusplus_data_stores_per_version_autoseg[percentage][1][:,0])

    print('interactive init \n')
    for percentage in deepeditplusplus_data_stores_per_version_deepgrow.keys():
        print(f'data partition configuration {percentage} \n')

        iters = [1,5,10]

        for iter in iters:
            print(f'comparing iter {iter} to init \n')
            wilcoxon_test(deepeditplusplus_data_stores_per_version_deepgrow[percentage][1][:,iter], deepeditplusplus_data_stores_per_version_deepgrow[percentage][1][:, 0])
