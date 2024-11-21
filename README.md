# PPGL_AI

## This repository is created to release the model weights for the following paper: "An Automated Pheochromocytoma and Paraganglioma lesion segmentation AI-model at whole-body 68Ga- DOTATATE PET/CT".
## Please cite if you are using the model: 
### Haque, F., Carrasquillo, J.A., Turkbey, E.B. et al. [An automated pheochromocytoma and paraganglioma lesion segmentation AI-model at whole-body 68Ga- DOTATATE PET/CT](https://ejnmmires.springeropen.com/articles/10.1186/s13550-024-01168-5). EJNMMI Res 14, 103 (2024).https://doi.org/10.1186/s13550-024-01168-5

### We have used 3D full resolution [nnUNet framework](https://www.nature.com/articles/s41592-020-01008-z). Follow the instrustion below to run inference on new dataset using our model.

### Insturctions: 
1. First, create a conda environment. You can name it to your liking; for example, ***'petct-env'***.
2. Install nnUNet. Installation process can be found in the following link: [documentation/installation_instructions.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
3. Create a new folder (anyname). Create the following three sub-folder in this directory: ***'nnUNet_raw'***, ***'nnUNet_preprocessed'***,***'nnUNet_results'*** inside the folder. The names should be exactly the same.
5. Create another folder ***"Dataset101_PETCT"*** inside ***'nnUNet_raw'***, ***'nnUNet_preprocessed'***,***'nnUNet_results'*** folders. This is important for nnUNet to identify which dataset to process.
6. nnU-Net expects datasets in a structured format. This format is inspired by the data structure of the Medical Segmentation Decthlon. Please read the following link for dataset conversion: [how-to-use-nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)
7. Image file should be in nifti format. USe the following package: [TCIA_processing](https://github.com/lab-midas/TCIA_processing) and use the following command to conver the images from DICOM to NIFTI:
  ````
    python3 -W ignore tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/
  ````
9.  PET images should be renamed as channel 1 input with ***'_0000.nii.gz'*** extension and CT images ***'_0001.nii.gz'***. Example PET image: ***PETCT_0ea07b421b_0000.nii.gz***, CT Image: ***PETCT_0ea07b421b_0001.nii.gz***
10. The PET/CT image files needs to be put inside the ***'/nnUNet_raw/Dataset101_PETCT/imagesTe'*** path.
11. ***"dataset_fingerprint.json"***, ***"nnUNetPlans.json"***,***"dataset.json"*** files should place inside ***"/nnUNet_preprocessed/Dataset101_PETCT"*** path.    
12. Model weights can be obtained by request only.
13. Plase the ***model weights*** inside the following path: ***"nnUNet_results\Dataset101_PETCT\nnUNetTrainer__nnUNetPlans__3d_fullres/"***. Inside ***'nnUNetTrainer__nnUNetPlans__3d_fullres'*** folder, model weights from **5 folds** in speparate folder be present. 
14. Once everything is set, run the bash file ***"inference.sh"*** to run inference using the model weights. Please modify the folder paths ***'nnUNet_raw'***, ***'nnUNet_preprocessed'***,***'nnUNet_results'*** according to your set up directories inside the *.sh* file.
