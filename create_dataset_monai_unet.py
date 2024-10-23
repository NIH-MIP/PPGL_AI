import pandas as pd
import glob
import os
import numpy as np
from RandomSplit.RandomSplit import MakeRandomSplit,RandomSplit

def get_train_test_val_split():
    Main_dir = '/media/air/18TBDATA/Fahmida_PETCT'
    meta_data= pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/Clinical Metadata FDG PET_CT Lesions.csv')
    meta_data= meta_data[["Subject ID", "diagnosis"]]
    meta_data= pd.read_csv('/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/Clinical Metadata FDG PET_CT Lesions.csv')
    print(meta_data.head(5))
    patients_id=pd.DataFrame(sorted(meta_data["Subject ID"].unique()))
    meta_data= meta_data[["Subject ID", "diagnosis"]]
    meta_data = meta_data.groupby(["Subject ID","diagnosis"])['diagnosis'].count()
    meta_data.to_excel("/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/output.xlsx")  
    data= meta_data.T
    W= np.array(data.iloc[:, 1:])
    column_map= data.columns[1:].tolist()

    print("Column map",len(column_map))
    p_train=0.70
    p_test=0.20
    train_list, test_list, val_list, res_train, res_test = MakeRandomSplit(W, p_train, p_test, column_map, tries=10)
    print("Splits", "Train", len(train_list), "Val", len(val_list), "Test", len(test_list))

    train_subject= {"Subject ID": train_list}
    train_subject = pd.DataFrame(train_subject)
    train_subject.to_excel("/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/train_list.xlsx")
                        
    val_subject= {"Subject ID": val_list}
    val_subject = pd.DataFrame(val_subject)
    val_subject.to_excel("/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/val_list.xlsx")

    test_subject= {"Subject ID":test_list}
    test_subject = pd.DataFrame(test_subject)
    test_subject.to_excel("/home/air/Shared_Drives/MIP_network/MIP/AIR/Projects/Project_OS_Kaplan/FDG_PET_CT_Image_database/test_list.xlsx")
    data= {"Subject ID": [], "diagnosis": []}
    for i in range(len(val_subject["Subject ID"])):
            for j in range(len(meta_data["Subject ID"])):
                if val_subject["Subject ID"][i] == meta_data["Subject ID"][j]:
                    # print(meta_data["diagnosis"][j])#if the country codes match
                    data["diagnosis"].append(meta_data["diagnosis"][j]) #gets the latitude of the matched country code
                    data["Subject ID"].append(meta_data["Subject ID"][j]) #gets the longitude

    # print(data)

    data = pd.DataFrame(data)        
    print(data.head(5))

def return_dictionary(data_dir, data_directory):
    images_pt= []
    images_ct= []
    label_name =[]
    
    for files in data_directory: 
        path= os.path.join(data_dir, files)
        for e in  os.listdir(path): 
            f = os.path.join(path, e)
            if os.path.exists(f):
                images_pt.append(os.path.join(f, "SUV.nii.gz"))
                images_ct.append(os.path.join(f, "CTres.nii.gz"))
                label_name.append(os.path.join(f, "SEG.nii.gz"))

    dicts = [{"image_pt": ''.join(image_name_pt), "image_ct": ''.join(image_name_ct), "label": ''.join(label_name)}
                for image_name_pt, image_name_ct, label_name in zip(images_pt, images_ct,label_name)]
    images_pt.clear()
    images_ct.clear()
    label_name.clear()
    return dicts

def get_data(data_dir):
    # data_dir = '/media/air/18TBDATA/Fahmida_PETCT/Image_formate_changed' 
    train_list=pd.read_excel("/data/MIP/fahmida_PETCT/Fahmida_PETCT/Unet_Model/train_list.xlsx")
    test_list=pd.read_excel("/data/MIP/fahmida_PETCT/Fahmida_PETCT/Unet_Model/test_list.xlsx")
    val_list=pd.read_excel("/data/MIP/fahmida_PETCT/Fahmida_PETCT/Unet_Model/val_list.xlsx")

    images_pt= []
    images_ct= []
    label_name =[]
    
    for files in train_list["Subject ID"]: 
        path= os.path.join(data_dir, files)
        for e in  os.listdir(path): 
            f = os.path.join(path, e)
            if os.path.exists(f):
                images_pt.append(os.path.join(f, "SUV.nii.gz"))
                images_ct.append(os.path.join(f, "CTres.nii.gz"))
                label_name.append(os.path.join(f, "SEG.nii.gz"))

    Train_dicts = [{"image_pt": ''.join(image_name_pt), "image_ct": ''.join(image_name_ct), "label": ''.join(label_name)}
                for image_name_pt, image_name_ct, label_name in zip(images_pt, images_ct,label_name)]
    images_pt.clear()
    images_ct.clear()
    label_name.clear()

    for files in test_list["Subject ID"]: 
        path= os.path.join(data_dir, files)
        for e in  os.listdir(path): 
            f = os.path.join(path, e)
            if os.path.exists(f):
                '''to do: one more loop throguh f folder as now it should contain multiple crop folders. 
                assuming creating multiple crop folder each containing pet_crop ct_crop, mask_crop'''
                images_pt.append(os.path.join(f, "SUV.nii.gz"))
                images_ct.append(os.path.join(f, "CTres.nii.gz"))
                label_name.append(os.path.join(f, "SEG.nii.gz"))

    Test_dicts = [{"image_pt": ''.join(image_name_pt), "image_ct": ''.join(image_name_ct), "label": ''.join(label_name)}
                for image_name_pt, image_name_ct, label_name in zip(images_pt, images_ct,label_name)]
    images_pt.clear()
    images_ct.clear()
    label_name.clear()
    
    for files in val_list["Subject ID"]: 
        path= os.path.join(data_dir, files)
        for e in  os.listdir(path): 
            f = os.path.join(path, e)
            if os.path.exists(f):
                images_pt.append(os.path.join(f, "SUV.nii.gz"))
                images_ct.append(os.path.join(f, "CTres.nii.gz"))
                label_name.append(os.path.join(f, "SEG.nii.gz"))

    Val_dicts = [{"image_pt": ''.join(image_name_pt), "image_ct": ''.join(image_name_ct), "label": ''.join(label_name)}
                for image_name_pt, image_name_ct, label_name in zip(images_pt, images_ct,label_name)]

    images_pt.clear()
    images_ct.clear()
    label_name.clear()
    return Train_dicts, Test_dicts, Val_dicts
