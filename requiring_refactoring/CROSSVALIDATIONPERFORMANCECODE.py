import os
import csv
import numpy as np 
import argparse

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
    
def wilcoxon_test_multi_iter_improvement(array_1, array_2):
    #we do it by iteration, so we extract the iter array to input to the function above

    #We are testing whether there is a statistical significance in the per-sample dice score changes between iter 0 and iter 1, iter 0 and iter 5 and iter 0 and iter 10
    #for the two approaches.

    #extracting initialisations:

    array_1_initialisation = array_1[:,0]
    array_2_initialisation = array_2[:,0]


    for i in [1,5,10]:
        print(f'iter {i} test \n')
        array_1_extract_col = array_1[:, i]
        array_2_extract_col = array_2[:, i]

        #find the difference since initialisation
        array_1_difference = array_1_extract_col - array_1_initialisation 
        array_2_difference = array_2_extract_col - array_2_initialisation 

        wilcoxon_test(array_1_difference, array_2_difference)

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
        assert num_samples == 200
        dice_scores_array = (dice_scores_array[0:num_samples, :] + dice_scores_array[num_samples:2*num_samples, :] + dice_scores_array[2*num_samples:, :])/3
        dice_score_names = dice_score_names[0:num_samples]
    
    return [dice_score_names, dice_scores_array]

def summary_extraction(array_dict):
    #assumes a dict which contains full names list which is N_samples long, and dice score arrays which are N_samples x N_iters: 
    output_dict = dict() 

    

    for dict_key in array_dict.keys():
        # dice_scores_names = array_dict[dict_key][0]
        dice_scores_array = array_dict[dict_key][1]

        # if repeat_bool == False:
        #     pass
        # else:  
        #     num_samples = int(dice_scores_array.shape[0]/3)
        #     dice_scores_array = (dice_scores_array[0:num_samples, :] + dice_scores_array[num_samples:2*num_samples, :] + dice_scores_array[2*num_samples:, :])/3            
        # #subdict which we will save into the full output dict
        output_dict_subdict = dict() 

        per_iter_minimum_array = np.min(dice_scores_array, axis=0)
        output_dict_subdict['per_iter_minimum_array'] = per_iter_minimum_array
        
        per_iter_lq_array = np.quantile(dice_scores_array, 0.25, axis=0)
        output_dict_subdict['per_iter_lq_array'] = per_iter_lq_array
        
        per_iter_median_array = np.quantile(dice_scores_array, 0.5, axis=0)
        output_dict_subdict['per_iter_median_array'] = per_iter_median_array
        

        #ADD THE PER ITER CHANGE TOO. 
        # per_iter_median_change_array = np.diff(per_iter_median_array)
        per_sample_diff_relative_to_init_per_iter = dice_scores_array - dice_scores_array[:,0].reshape(dice_scores_array[:,0].shape[0],1)
        output_dict_subdict['per_sample_per_iter_diff_relative_to_init_mean_array'] = np.mean(per_sample_diff_relative_to_init_per_iter, axis=0)

        output_dict_subdict['per_sample_per_iter_diff_relative_to_init_median_array'] = np.median(per_sample_diff_relative_to_init_per_iter, axis=0)

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

        # #ADD THE PER ITER CHANGE TOO. 
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


        with open(os.path.join(output_path, 'summary_across_folds.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(dataset_snapshot_name)

        snapshot_summary_dict = output_dict[folds_key]

        for summary_statistic_key in snapshot_summary_dict.keys():
            summary_stat_row = [summary_statistic_key]
            for score in snapshot_summary_dict[summary_statistic_key]:
                summary_stat_row.append(score) 

            with open(os.path.join(output_path, 'summary_across_folds.csv'),'a') as f:
                
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

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='BraTS2021_01620.nii.gz')
    args = parser.parse_args()


    root_folder = '/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/validation_results/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT'
    image_name = args.image_name #"BraTS2021_01620.nii.gz" 
    epoch = 'best_val_score_epoch'

    deepediting_infer_runs = 'run_0_1_2'

    autoseg_infer_runs = 'run_0'
    
    root_folder_deepedit = os.path.join(root_folder, 'deepedit')
    root_folder_deepedit_modality_change = os.path.join(root_folder, 'deepeditModalityChange')
    root_folder_deepeditplus = os.path.join(root_folder,'deepeditplus')
    root_folder_deepeditplusplus = os.path.join(root_folder, 'deepeditplusplus')
    root_folder_nnunet = os.path.join(root_folder, 'nnUnet')

    deepedit_dataset_versions = ['23072024_140322']
    deepedit_modality_change_dataset_versions = ['23072024_140327']
    deepeditplus_dataset_versions = ['10072024_201348']#, '12072024_010437', '12072024_234336', '13072024_125938']
    deepeditplusplus_dataset_versions = ['10072024_182402']#, '12072024_164436', '14072024_022318', '15072024_010034'] #New version is going to be this instead ['10072024_182402', '12072024_164436', '18072024_004553', '15072024_010034'] 

    #when nnunet versions are done training then change this back. 

    nnunet_dataset_versions = ['nnUnet_fold_1_2_3_4']#, 'nnUnet_fold_1', 'nnUnet_fold_1', 'nnUnet_fold_1']#['nnUnet_fold_1_2_3_4', 'nnUnet_fold_1_2_3', 'nnUnet_fold_1_2', 'nnUnet_fold_1']


    ### string which encodes what the task is:

    # autoseg_only_runtask = 'imagesTs_autoseg'

    autoseg_only_runtask = 'autoseg'

    autoseg_init_runtask = 'deepedit_autoseg_initialisation_numIters_10'


    # interactive_init_runtask = 'deepedit_deepgrow_initialisation_numIters_10'

    #we are going to save in order from largest to smallest dataset partition

    deepedit_data_stores_per_version_autoseg = dict()

    deepedit_modality_change_data_stores_per_version_autoseg = dict()


    deepeditplus_data_stores_per_version_autoseg = dict()

    # deepeditplus_data_stores_per_version_deepgrow = dict()



    deepeditplusplus_data_stores_per_version_autoseg = dict()

    # deepeditplusplus_data_stores_per_version_deepgrow = dict()


    nnunet_data_stores_per_version_autoseg = dict()


    # folds_list_names = ['100%', '75%', '50%', '25%']
    folds_list_names = ['train_folds_1_2_3_4_validation_fold_0'] #['val_0_train_1_2_3_4', 'val_1_train_0_2_3_4', 'val_2_train_0_1_3_4', 'val_3_train_0_1_2_4', 'val_4_train_0_1_2_3']

    #we conjoin the fold list names with the dataset versions so we send the file to the corresponding spot.
    
    # #deepedit first

    # fold_model_version_datetime_conjoined = dict()

    # for index, model_version in enumerate(deepedit_dataset_versions):

    #     fold_list = folds_list_names[index]
    #     fold_model_version_datetime_conjoined[fold_list] = model_version


    # #deepeditplusplus
    # for index, model_version in enumerate(deepedit_modality_change_dataset_versions):

    #     fold_list = folds_list_names[index]
    #     fold_model_version_datetime_conjoined[fold_list] = model_version

    # fold_model_version_datetime_conjoined = dict()

    # for index, model_version in enumerate(deepeditplus_dataset_versions):

    #     fold_list = folds_list_names[index]
    #     fold_model_version_datetime_conjoined[fold_list] = model_version


    # #deepeditplusplus
    # for index, model_version in enumerate(deepeditplusplus_dataset_versions):

    #     fold_list = folds_list_names[index]
    #     fold_model_version_datetime_conjoined[fold_list] = model_version


    # #nnunet 
    # for index, model_version in enumerate(nnunet_dataset_versions):

    #     fold_list = folds_list_names[index]
    #     fold_model_version_datetime_conjoined[fold_list] = model_version



    

    #extract the scores for each ablation
    for index, version in enumerate(deepeditplus_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepeditplus, folds_list_names[index], autoseg_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # results_path_deepgrow_init = os.path.join(root_folder_deepeditplus, interactive_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # print(folds_list_names[index])
        # print(results_path_autoseg_init)
        # print(results_path_deepgrow_init)
        deepeditplus_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=True)
        # deepedit_data_stores_per_version_deepgrow[folds_list_names[index]] = dice_score_reformat(dice_score_extraction(results_path_deepgrow_init))


    for index,version in enumerate(deepeditplusplus_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepeditplusplus, folds_list_names[index], autoseg_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # results_path_deepgrow_init = os.path.join(root_folder_deepeditplusplus, interactive_init_runtask, version, epoch, 'run_collection', deepediting_infer_runs)
        # print(folds_list_names[index])
        # print(results_path_autoseg_init)
        # print(results_path_deepgrow_init)
        deepeditplusplus_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=True)
        # deepeditplusplus_data_stores_per_version_deepgrow[folds_list_names[index]] = dice_score_reformat(dice_score_extraction(results_path_deepgrow_init))

    for index,version in enumerate(nnunet_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_nnunet, folds_list_names[index], autoseg_only_runtask, version, epoch, 'run_collection', autoseg_infer_runs)
        nnunet_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=False)
        # deepeditplusplus_data_stores_per_version_deepgrow.append(dice_score_extraction(results_path_autoseg_init))


    for index,version in enumerate(deepedit_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepedit, folds_list_names[index],  autoseg_only_runtask, version, epoch, 'run_collection', autoseg_infer_runs)
        deepedit_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=False)
    
    for index,version in enumerate(deepedit_modality_change_dataset_versions):
        results_path_autoseg_init = os.path.join(root_folder_deepedit_modality_change, folds_list_names[index], autoseg_only_runtask, version, epoch, 'run_collection', autoseg_infer_runs)
        deepedit_modality_change_data_stores_per_version_autoseg[folds_list_names[index]] = repeating_run_averaging(dice_score_reformat(dice_score_extraction(results_path_autoseg_init)), repeat_bool=False)

    #the data is saved as an N_samples x N_iters array, here we extract the scores which we will use going forward? we are going to extract means, medians, iqr, min and max.

    summary_scores_deepeditplus_autoseg_init = summary_extraction(deepeditplus_data_stores_per_version_autoseg)
    # summary_scores_deepedit_deepgrow_init = summary_extraction(deepedit_data_stores_per_version_deepgrow)

    summary_scores_deepeditplusplus_autoseg_init = summary_extraction(deepeditplusplus_data_stores_per_version_autoseg)
    # summary_scores_deepeditplusplus_deepgrow_init = summary_extraction(deepeditplusplus_data_stores_per_version_deepgrow)

    summary_scores_nnunet_autoseg_only =  summary_extraction(nnunet_data_stores_per_version_autoseg) 

    summary_scores_deepedit_autoseg_only = summary_extraction(deepedit_data_stores_per_version_autoseg)

    summary_scores_deepedit_modality_change_autoseg_only = summary_extraction(deepedit_modality_change_data_stores_per_version_autoseg)

    ######################################################################################################################


    # saving the outputs to a csv file which contains all the summaries across all folders....

    #model versions conjoined 
    deepedit_model_versions_conjoined = '_'.join(deepedit_dataset_versions)
    deepedit_modality_change_model_versions_conjoined = '_'.join(deepedit_modality_change_dataset_versions)
    nnunet_model_versions_conjoined = '_'.join(nnunet_dataset_versions)
    deepeditplus_model_versions_conjoined = '_'.join(deepeditplus_dataset_versions)
    deepeditplusplus_model_versions_conjoined = '_'.join(deepeditplusplus_dataset_versions)




    root_summaries_folder = '/home/parhomesmaeili/MonaiLabel Deepedit++ Development/MONAILabelDeepEditPlusPlus/validation_results_summaries/validation_spreadsheet_summaries/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT'
    # image_name = args.image_name #"BraTS2021_01620.nii.gz"
    os.makedirs(root_summaries_folder, exist_ok=True)
    
    epoch = 'best_val_score_epoch'

    deepediting_infer_runs = 'run_0_1_2'

    autoseg_infer_runs = 'run_0'

    summary_root_folder_deepedit = os.path.join(root_summaries_folder,'deepedit')
    summary_root_folder_deepedit_modality_change = os.path.join(root_summaries_folder,'deepedit_modality_change')
    summary_root_folder_deepeditplus = os.path.join(root_summaries_folder,'deepeditplus')
    summary_root_folder_deepeditplusplus = os.path.join(root_summaries_folder, 'deepeditplusplus')
    summary_root_folder_nnunet = os.path.join(root_summaries_folder, 'nnUnet')


    autoseg_only_runtask = 'autoseg'

    autoseg_init_runtask = 'deepedit_autoseg_initialisation_numIters_10'


    task_root_folder_deepedit_autoseg = os.path.join(summary_root_folder_deepedit)

    task_root_folder_deepedit_modality_change_autoseg = os.path.join(summary_root_folder_deepedit_modality_change)

    task_root_folder_nnunet_autoseg = os.path.join(summary_root_folder_nnunet, autoseg_only_runtask)

    ### aggregating by the inference task runs
    task_root_folder_deepeditplus_autoseg_init = os.path.join(summary_root_folder_deepeditplus, autoseg_init_runtask)
    # task_root_folder_deepedit_deepgrow_init = os.path.join(summary_root_folder_deepedit, interactive_init_runtask)

    task_root_folder_deepeditplusplus_autoseg_init = os.path.join(summary_root_folder_deepeditplusplus, autoseg_init_runtask)
    # task_root_folder_deepeditplusplus_deepgrow_init = os.path.join(summary_root_folder_deepeditplusplus, interactive_init_runtask)

    


    #aggregating by the model versions (across the dataset versions) for all of those inference task runs
    model_versions_aggregate_save_root_path_deepedit_autoseg = os.path.join(task_root_folder_deepedit_autoseg, deepedit_dataset_versions[0])

    model_versions_aggregate_save_root_path_deepedit_modality_change_autoseg = os.path.join(task_root_folder_deepedit_modality_change_autoseg, deepedit_modality_change_dataset_versions[0])
    
    model_versions_aggregate_save_root_path_nnunet_autoseg = os.path.join(task_root_folder_nnunet_autoseg, nnunet_dataset_versions[0])

    model_versions_aggregate_save_root_path_deepeditplus_autoseg_init = os.path.join(task_root_folder_deepeditplus_autoseg_init, deepeditplus_dataset_versions[0])
    # model_versions_aggregate_save_root_path_deepedit_deepgrow_init = os.path.join(task_root_folder_deepedit_deepgrow_init, deepedit_model_versions_conjoined)

    model_versions_aggregate_save_root_path_deepeditplusplus_autoseg_init = os.path.join(task_root_folder_deepeditplusplus_autoseg_init, deepeditplusplus_dataset_versions[0])
    # model_versions_aggregate_save_root_path_deepeditplusplus_deepgrow_init = os.path.join(task_root_folder_deepeditplusplus_deepgrow_init, deepeditplusplus_model_versions_conjoined)

######################################################################################################################################################################

    #pointing path to epoch/infer run collection



    epoch_path_deepedit = os.path.join(model_versions_aggregate_save_root_path_deepedit_autoseg, epoch)

    run_collection_path_deepedit = os.path.join(epoch_path_deepedit, 'run_collection', autoseg_infer_runs)



    #pointing path to epoch/infer run collection



    epoch_path_deepedit_modality_change = os.path.join(model_versions_aggregate_save_root_path_deepedit_modality_change_autoseg, epoch)

    run_collection_path_deepedit_modality_change = os.path.join(epoch_path_deepedit_modality_change, 'run_collection', autoseg_infer_runs)




    #pointing path to epoch/infer run collection



    epoch_path_nnunet = os.path.join(model_versions_aggregate_save_root_path_nnunet_autoseg, epoch)

    run_collection_path_nnunet = os.path.join(epoch_path_nnunet, 'run_collection', autoseg_infer_runs)





    #pointing path to epoch/infer run collection
    epoch_path_deepeditplus_autoseg_init = os.path.join(model_versions_aggregate_save_root_path_deepeditplus_autoseg_init, epoch)
    

    run_collection_path_deepeditplus_autoseg_init = os.path.join(epoch_path_deepeditplus_autoseg_init, 'run_collection', deepediting_infer_runs)
    





    #pointing path to epoch/infer run collection
    epoch_path_deepeditplusplus_autoseg_init = os.path.join(model_versions_aggregate_save_root_path_deepeditplusplus_autoseg_init, epoch)
    

    run_collection_path_deepeditplusplus_autoseg_init = os.path.join(epoch_path_deepeditplusplus_autoseg_init, 'run_collection', deepediting_infer_runs)




    print(run_collection_path_deepedit)
    print(run_collection_path_deepedit_modality_change)
    print(run_collection_path_nnunet)
    print(run_collection_path_deepeditplus_autoseg_init)
    # print(run_collection_path_deepedit_deepgrow_init)
    print(run_collection_path_deepeditplusplus_autoseg_init)
    # print(run_collection_path_deepeditplusplus_deepgrow_init)

    save_summarised(run_collection_path_deepedit, summary_scores_deepedit_autoseg_only)
    save_summarised(run_collection_path_deepedit_modality_change, summary_scores_deepedit_modality_change_autoseg_only)
    save_summarised(run_collection_path_nnunet, summary_scores_nnunet_autoseg_only)
    save_summarised(run_collection_path_deepeditplus_autoseg_init, summary_scores_deepeditplus_autoseg_init)
    # save_summarised(run_collection_path_deepedit_deepgrow_init, summary_scores_deepedit_deepgrow_init)
    save_summarised(run_collection_path_deepeditplusplus_autoseg_init, summary_scores_deepeditplusplus_autoseg_init)
    # save_summarised(run_collection_path_deepeditplusplus_deepgrow_init, summary_scores_deepeditplusplus_deepgrow_init)


    #testing statistical significance of our results
    print('statistical test for modality correction \n')
    wilcoxon_test(deepedit_data_stores_per_version_autoseg[folds_list_names[0]][1].flatten(), deepedit_modality_change_data_stores_per_version_autoseg[folds_list_names[0]][1].flatten())

    print('statistical test for the padding modification \n')
    wilcoxon_test(deepedit_modality_change_data_stores_per_version_autoseg[folds_list_names[0]][1].flatten(), deepeditplus_data_stores_per_version_autoseg[folds_list_names[0]][1][:,0][:200])

    print('statistical test for the per_iteration performance for inner loop modification etc \n')

    wilcoxon_test_multi_iter_improvement(deepeditplus_data_stores_per_version_autoseg[folds_list_names[0]][1], deepeditplusplus_data_stores_per_version_autoseg[folds_list_names[0]][1])

    print('statistical test for the raw dice scores at different iterations...')

    wilcoxon_test_multi(deepeditplus_data_stores_per_version_autoseg[folds_list_names[0]][1], deepeditplusplus_data_stores_per_version_autoseg[folds_list_names[0]][1])