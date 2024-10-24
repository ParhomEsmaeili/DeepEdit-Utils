import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/validation")
    parser.add_argument("--datetime", nargs="+", default=["31052024_195641"])
    parser.add_argument("--checkpoint", nargs="+", default=["best_val_score_epoch"])
    parser.add_argument("--infer_run", nargs="+", default=['0', '1', '2'])
    parser.add_argument("-ta", "--task", nargs="+", default=["deepedit", "autoseg", "3"], help="The framework selection + subtask/mode which we want to execute")
    parser.add_argument("--app_dir", default = "DeepEditPlusPlus Development/DeepEditPlusPlus")
    parser.add_argument("--job", default= "collecting")
    
    
    args = parser.parse_args()