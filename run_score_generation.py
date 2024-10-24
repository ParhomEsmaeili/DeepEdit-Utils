from test_score_generation import test_scores 

if __name__ == '__main__':

    args = dict() 
    args['studies'] = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT" #The name of the dataset which contains all of the images, segmentations, class configs etc.
    args['datetime'] = "31052024_195641"  #The name of the model datetime which is under consideration
    args['checkpoint'] = "best_val_score_epoch" #The name of the epoch of the model datetime which has been used for inference.
    args["inference_run_num"] = '0'  #The number of the inference run which is under consideration (for probabilistic simulation of clicks at inference)
    args['inference_run_parametrisation'] = dict() 
    
    #The parametrisation of the clicks at inference, could be "None", could be "Fixed Click Size" parametrisation (should have the values provided in a list) or it could be "Dynamic Click Size"
    #This information is presented in dict format, the key should have the type of parametrisation, the value should have the parametrisation values (if appropriate, only for the fixed click size)
    
    #Also a parametrisation of the click collection strategy, i.e. whether it is CIM or 1-Iter. SIM.
    #  
    args['click_weightmap_dict'] = dict() 
    #The dict of click-based weightmap types and their parametrisations which are applied for the generation of the mask in metric computation, e.g. ellipsoid, scaled euclidean etc.

    args['gt_weightmap_types'] = []
    #The list of the click-based weightmap types (non-parametric), e.g. connected component, or none.

    args['base_metric'] = 'Dice'
    # The base metric being used for computation of the metric scores

    args['human_measure'] = 'None'
    #The human measure which is being for metric mask-generation, e.g. local responsiveness.

    args['inference_run_mode'] = ['Editing', 'Autoseg', '10'] # The inference run mode which we want to perform score computation for, if it is just an initiatlisation then this is just one item long.
    
    args['app_dir'] = 'DeepEditPlusPlus Development/DeepEditPlusPlus' #The path to the app directory from the base/home directory.

    args['include_background_mask'] = True #The bool which determines whether we use the background points for generating the weighting mask

    args['include_background_metric'] = False #The bool which determines whether we use the background class for generating and outputting, multi-class and per class scores. 
    
    args['ignore_empty'] = True #The bool which determines whether we ignore the scores for instances where there is no denominator (because there was no ground truth) 

    args['per_class_scores'] = True #The bool which determines whether we generate multi-class AND per class scores, or not.
    
    args['sequentiality_mode'] = 'SIM' #The argument which determines whether we are working with CIM or SIM based models (even for the default score computations we just assume SIM)
    
    args['dataset_subset'] = 'validation' #The argument which determines whether we are computing scores for the validation outputs, or for the test segmentation outputs. 

    score_generator = test_scores(args)

    score_generator() 


    

