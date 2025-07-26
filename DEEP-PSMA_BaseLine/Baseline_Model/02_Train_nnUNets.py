import os
from os.path import join,isdir
import SimpleITK as sitk
import shutil
import pandas as pd


import nnunet_config_paths #sets and exports paths for nnU-Net configuration

##Script to facilitate calling nnUNetv2_train for different dataset/fold combinations.
##Simple logic to test if a tracer/fold has completed but not suitable to run in parallel.
##Much better to queue jobs and run "nnUNetv2_train 801/802 3d_fullres 0/1/2/3/4" if familiar.
##If run separately, export nn raw/preprocessed/results locations to match the config ./data/nnUNet_data

##If running this script and "nnUNetv2_train" is not in PATH location, adjust "os.system(..." line to match appropriate path

dataset_dict=nnunet_config_paths.dataset_dictionary #get dictionary of tracer names + task numbers

#syntax: nnUNetv2_train 801 3d_fullres 1

for tracer in list(dataset_dict): #iterate through both tracers ['PSMA','FDG']
    task_number=str(dataset_dict[tracer]) #get nnUNet task #
    dataset_name='Dataset'+task_number+'_'+tracer+'_PET'
##    for fold in ['0','1','2','3','4']: #iterate through the 5 nnunet folds
    for fold in ['0']: 
        final_result_fname=join(nnunet_config_paths.nn_results_dir,dataset_name,'nnUNetTrainer__nnUNetPlans__3d_fullres','fold_'+fold,'checkpoint_final.pth')
        print('checking for', final_result_fname) #check if training is completed previously
        if not os.path.exists(final_result_fname):
            print('not found, running nnUNetv2_train')
            os.system('nnUNetv2_train '+task_number+' 3d_fullres '+fold) #+' --c' nnunet training command
        else:
            print('final training model detected. skipping to next tracer/fold iteration')
