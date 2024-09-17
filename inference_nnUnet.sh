#!/bin/bash


export nnUNet_raw="path-to-folder/nnUNet_raw"
export nnUNet_preprocessed="path-to-folder/nnUNet_preprocessed"
export nnUNet_results="path-to-folder/nnUNet_results"


nnUNetv2_predict -d DATASET_ID -i $nnUNet_raw/DATASET_ID/imagesTe/ -o $nnUNet_results/DATASET_ID/nnUNetTrainer__nnUNetPlans__3d_fullres/Test_Set/ -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans




