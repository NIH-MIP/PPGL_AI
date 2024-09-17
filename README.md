# PPGL_AI

This repository is created to release the model weights for the following paper: "An Automated Pheochromocytoma and Paraganglioma lesion segmentation AI-model at whole-body 68Ga- DOTATATE PET/CT"

We have used 3D full resolution nnUNet model( https://www.nature.com/articles/s41592-020-01008-z) and the model was developed using the same instruction as by the nnUNet repository (https://github.com/MIC-DKFZ/nnUNet).

Model weights can be downloaded from google drive: https://drive.google.com/drive/folders/1Id9BF5YhHBD_qq52p0EBfYt-y0Y8sPvb?usp=sharing
Also download the dataset.json, dataset_fingerprint.json, nnUNetPlans.json files and keep them in the "nnUNet_preprocessed" folder.

Use inference.sh to run inference using the model weights. The folders and data structure as same as instructed for nnUNet.
