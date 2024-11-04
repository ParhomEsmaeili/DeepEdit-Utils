import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse 
import csv

def validation_score_computation(file_path, num_epochs):
    #Loading the dataframes into the system:
    with open(file_path, newline='') as csvfile:

        dataset = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        new_dataset = []

        for index, row in enumerate(dataset):
            if index == 0:
                metric_names = row[0].split(',')
            else:
                row_split = row[0].split(',')
                sub_list = [float(i) for i in row_split]
                new_dataset.append(sub_list)
        
        num_val_images = len(new_dataset)/num_epochs 
        num_columns = len(new_dataset[0])
        
        final_datasets = []
        for i in range(num_columns):
            sub_list = []
            for j in range(0, num_epochs):
                tmp_list = []
                for index in range(int(num_val_images)):
                    tmp_list.append(new_dataset[j * int(num_val_images) + index][i])
                
                val_score_average = sum(tmp_list)/len(tmp_list)
                sub_list.append((j + 1, val_score_average))
            final_datasets.append(sub_list)
        print(final_datasets)
        return metric_names, final_datasets

def parse_arguments():
    parser = argparse.ArgumentParser("External Validation Score Processing")
    parser.add_argument("--datetime", default="20241103_142602")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--num_epochs", default='300')
    return parser






if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    
    model_version_folder = os.path.join(args.datetime, args.model_dir)

    # print(type(args.num_epochs))

    app_dir = os.path.join(os.path.expanduser('~'), 'DeepEditPlusPlus Development', 'DeepEditPlusPlus', 'external_validation')
    file_directory = os.path.join(app_dir, model_version_folder, 'validation_scores')
    metric_names, val_scores = validation_score_computation(os.path.join(file_directory, 'validation.csv'), int(args.num_epochs))

    logdir_path = os.path.join(file_directory, 'tensorboard_files')
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)
    writer = SummaryWriter(log_dir=logdir_path)

    for index, metric_list in enumerate(val_scores):
        metric_name = metric_names[index]
        for epoch, score in metric_list:
            # print(epoch, score)
            # # print(type(epoch))
            # # print(type(score))
            # print(metric_name)
            writer.add_scalar(metric_name + '_val', score, epoch)




    #writer = SummaryWriter() 
