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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/validation")
    parser.add_argument("--datetime", nargs="+", default=["31052024_195641"])
    parser.add_argument("--checkpoint", nargs="+", default=["best_val_score_epoch"])
    parser.add_argument("--infer_run", nargs="+", default=['0'])
    parser.add_argument("-ta", "--task", nargs="+", default=["deepedit", "autoseg", "3"], help="The framework selection + subtask/mode which we want to execute")
    parser.add_argument("--app_dir", default = "DeepEditPlusPlus Development/DeepEditPlusPlus")
    parser.add_argument("--job", default= "compute")
    
    args = parser.parse_args()

    app_dir = os.path.join(up(up(up(os.path.abspath(__file__)))), args.app_dir)
    framework = args.task[0]
    inference_task = args.task[1]
    
    dataset_name = args.studies[:-9]
    dataset_subset = args.studies[-8:]

    job = args.job