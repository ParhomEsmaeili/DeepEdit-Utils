import os 
import sys
from os.path import dirname as up
utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_dir)
from Score_Generation_And_Processing.statistical_significance_script import statistical_significance_assessment as stat_signif 

if __name__ == '__main__':

    args = dict() 
    args['studies'] = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised" #The name of the dataset which contains all of the images, segmentations, class configs etc.
    args['datetimes'] = ["20241102_121843", "20241104_135136"]  #The name of the model datetimes which are under consideration, OR the nnU-net model name, for example. 

    args['checkpoints'] = ["best_val_score_epoch", "best_val_score_epoch"] #The name of the epoch of the model datetime which has been used for inference.
    args["inference_run_nums"] = ['0','1','2']  #The list of the inference run nums which are under consideration (after being sample averaged across these)
    args['inference_run_parametrisation'] = {
        "None":["None"]
    } 
    
    #The parametrisation of the clicks at inference, could be "None", could be "Fixed Click Size" parametrisation (should have the values provided in a list) or it could be "Dynamic Click Size"
    #This information is presented in dict format, the key should have the type of parametrisation, the value should have the parametrisation values (if appropriate, only for the fixed click size)
    
    #The value must be a list! 
 
    args['click_weightmap_dict'] = {
        # "None":["None"]
        "Exponentialised Scaled Euclidean Distance":[1,1,1,1]
    } 

    #The dict of click-based weightmap types and their parametrisations which are applied for the generation of the mask in metric computation, e.g. ellipsoid, scaled euclidean etc.
    #The value must always be a list! 

    args["simulation_type"] = 'probabilistic' #The param which controls whether the simulation of the click was probabilistic or deterministic. 

    args['gt_weightmap_types'] = ["None"]
    #The list of the click-based weightmap types (non-parametric), e.g. connected component, or none.

    # args['base_metric'] = 'Dice'
    args['base_metric'] = 'Error Rate'

    # The base metric being used for computation of the metric scores

    args['derived_metric'] = ['Default',
                            # 'Relative To Init Score', 
                            # 'Per Iter Improvement Score'
                            ] 
    #Any derived metric which is being computed also, e.g. relative improvement in standard dice score.

    # args['human_measure'] = 'None'
    # args['human_measure'] = 'Local Responsiveness'
    args['human_measure'] = 'Temporal Non Worsening'
    
    #The human measure which is being for metric mask-generation, e.g. local responsiveness.

    args['inference_run_mode'] = ['Editing', 'Autoseg', '10'] # The inference run mode which we want to perform score computation for, if it is just an initiatlisation then this is just one item long.
    
    args['app_dir'] = 'DeepEditPlusPlus Development/DeepEditPlusPlus' #The path to the app directory from the base/home directory.

    args['include_background_metric'] = False #The bool which determines whether we used the background class for generating and outputting, multi-class and per class scores. 
    
    args['dataset_subset'] = 'validation' #The argument which determines whether we are computing scores for the validation outputs, or for the test segmentation outputs. 

    # args['include_nan'] = False #The argument which determines whether nans should be used in summarisation/dropped out (obviously not)

    # args['num_samples'] = 200 #The argument which controls the number of samples that are being used for score summarisation (e.g just the first N samples)

    # args['total_samples'] = 200 #The argument which controls the maximum number of total samples that could be available to be used for score summarisation 

    args['per_class_scores'] = False # Whether it should be applied for each class or not. We almost always have this equal to false because we only really care about the
    #cross-class score. The per-class scores are just there for a sanity check. 
    
    # #TODO: However, we should consider that we may want to implement this for the pure dice score!?


    args['statistical_test'] = {
        'Wilcoxon Signed Rank Test': {'p_value':0.05},
        # 'Paired T Test': {'p_value':0.05} 
    }
    
    #The argument (dict) which contains the information about which statistical tests to perform. Allows for any parametrisation required (e.g. the significance level)
    #

    stat_signif_class = stat_signif(args)

    stat_signif_class() 
