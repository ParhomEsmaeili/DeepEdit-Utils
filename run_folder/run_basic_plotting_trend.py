import os 
import sys
from os.path import dirname as up
utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(utils_dir)
from Score_Plotting.basic_plotting_trend import plot_trend as plotter

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
        # "Exponentialised Scaled Euclidean Distance":[1,1,1,1]
        'Ellipsoid':[5,5,5]
    } 

    #The dict of click-based weightmap types and their parametrisations which are applied for the generation of the mask in metric computation, e.g. ellipsoid, scaled euclidean etc.
    #The value must always be a list! 

    args["simulation_type"] = 'probabilistic' #The param which controls whether the simulation of the click was probabilistic or deterministic. 

    args['gt_weightmap_types'] = ["None"]
    #The list of the click-based weightmap types (non-parametric), e.g. connected component, or none.

    args['base_metric'] = 'Dice'
    # args['base_metric'] = 'Error Rate'

    # The base metric being used for computation of the metric scores

    args['derived_metric'] = ['Default',
                            ]
    
    # args['human_measure'] = 'None'
    args['human_measure'] = 'Local Responsiveness'
    # args['human_measure'] = 'Temporal Non Worsening'
    # args['human_measure'] = 'Temporal Consistency'

    #The human measure which is being for metric mask-generation, e.g. local responsiveness.

    args['inference_run_mode'] = ['Editing', 'Autoseg', '10'] # The inference run mode which we want to perform score computation for, if it is just an initiatlisation then this is just one item long.
    
    args['app_dir'] = 'DeepEditPlusPlus Development/DeepEditPlusPlus' #The path to the app directory from the base/home directory.

    # args['include_background_metric'] = True #The bool which determines whether we used the background class for generating and outputting, multi-class and per class scores. 
    
    args['dataset_subset'] = 'validation' #The argument which determines whether we are computing scores for the validation outputs, or for the test segmentation outputs. 

    # args['per_class_scores'] = False # Whether it should be applied for each class or not. We almost always have this equal to false because we only really care about the
    #cross-class score. The per-class scores are just there for a sanity check. 
    
    args['summary_dict'] = {
        'Mean': None, 
        'Median': None, 
        'Standard Deviation': None,
        # 'Interquartile Range': None,
        'Lower Quartile': None,
        'Upper Quartile': None,
        # 'Minimum' : None,
        # 'Maximum': None 
    }

    args['plot_info'] = {
        'legends':{
            "20241102_121843": 'Baseline Model',
            "20241104_135136": 'Approximate Loop Unrolling'
        }
    }

    #The argument (dict) which contains the information about which statistical tests to perform. Allows for any parametrisation required (e.g. the significance level)
    #


    stat_signif_class = plotter(args)

    stat_signif_class() 
