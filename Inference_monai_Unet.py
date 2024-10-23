import os
import torch
import shutil
import tempfile
import time
from datetime import datetime
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.transforms.utils import ndimage
from monai.utils import first
from tqdm import tqdm
from monai.handlers.utils import from_engine
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    ConcatItemsd, EnsureTyped,
    AsDiscrete,

)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
import SimpleITK as sitk
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
)
import torch
import glob
import pandas as pd
from datasetcheck import get_data
from scipy import ndimage
# from utils.utils import dice, resample_3d

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

#
def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("using cuda:0")
else:
    raise RuntimeError("this tutorial is intended for GPU, but no CUDA device is available")


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

''' Set learning rate, number of epochs, model name, sliding window size'''
lr = 1e-2
numephs = 1000
window_size = [128,128,32]
modelname = 'UNet_128_128_32' #set model name


# Setup output directory directory
main_dir='path-to-output-directory'
modelpath= os.path.join(main_dir, 'models')
outpath= os.path.join(main_dir, 'results')
if not os.path.exists(outpath):
    os.makedirs(outpath)
testpath= os.path.join(outpath, 'Test_Preds')
if not os.path.exists(testpath):
    os.makedirs(testpath)
testpath_gt= os.path.join(outpath, 'Test_GT')
if not os.path.exists(testpath_gt):
    os.makedirs(testpath_gt)
testpath_image= os.path.join(outpath, 'Test_PET')
if not os.path.exists(testpath_image):
    os.makedirs(testpath_image)
valpath_gt= os.path.join(outpath, 'VAL_GT')
if not os.path.exists(valpath_gt):
    os.makedirs(valpath_gt)
valpath_image= os.path.join(outpath, 'VAL_PET')
if not os.path.exists(valpath_image):
    os.makedirs(valpath_image)
valpath= os.path.join(outpath, 'Val_Preds')
if not os.path.exists(valpath):
    os.makedirs(valpath)

'''Data folder provided'''
data_dir = 'Path-to-dataset-folder'

'''load image files names: New way to split data stratified way.'''

Train_dicts, Test_dicts, Val_dicts= get_data(data_dir)

val_transforms = Compose([
        LoadImaged(keys=["image_pt", "image_ct", "label"]),
        EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
        ScaleIntensityRanged(keys=["image_ct"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
        ScaleIntensityRanged(keys=["image_pt"], a_min=0, a_max=15,b_min=0.0, b_max=1.0, clip=False),
        Spacingd(keys=["image_pt","image_ct", "label"], pixdim=(2, 2, 3), mode=("bilinear", "bilinear", "nearest")),
        # SpatialPadd(keys=["image_pt","image_ct", "label"], spatial_size=window_size),
        # EnsureTyped(keys=["image_pt","image_ct", "label"],device=device, track_meta=False),
        # RandCropByLabelClassesd(keys=["image_pt", "image_ct", "label"], label_key="label", spatial_size=window_size, ratios=[2, 1],num_classes=2, num_samples=6),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),  # concatenate pet and ct channels
        EnsureTyped(keys=["image_petct", "label"]),
])

print("hello!")


check_val = CacheDataset(data=Val_dicts, transform= val_transforms, cache_rate=0.1, num_workers=2, copy_cache=False)
val_loader = ThreadDataLoader(check_val, batch_size=1, num_workers=0)



model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
epoch_times = []
total_start = time.time()



'''inference on val set'''

print("Starting Interface")

path= os.path.join(modelpath, "UNet_crops_best_epoch-26.pth")
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


model.eval()
model.to(device)
test_count=0
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

int_time_S=datetime.fromtimestamp(time.time())
metric_val=[]
with torch.no_grad():
    for i,  val_data in enumerate(val_loader):
        test_count+=1
        val_inputs, val_labels = (val_data["image_petct"].to(device),  val_data["label"].to(device))
        original_affine = val_data["label_meta_dict"]["affine"][0].numpy()
        print(original_affine)
        # print(val_labels.shape)
        _, _, h, w, d = val_labels.shape
        target_shape = (h, w, d)
        print(target_shape)
        a,b =  val_data["image_petct"].meta["filename_or_obj"][0].split("/")[7], val_data["image_petct"].meta["filename_or_obj"][0].split("/")[8]
        img_name= a+"_"+b[:10]
        print( val_data["image_petct"].meta["filename_or_obj"][0].split("/")[6], val_data["image_petct"].meta["filename_or_obj"][0].split("/")[7], img_name)
        print("Inference on case {}".format(img_name))
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_inputs, window_size, sw_batch_size, model, overlap= 0.8,device=device)
        val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
        val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
        print("val_outputs ", val_outputs.shape, val_outputs.dtype)
        image_pt = sitk.ReadImage(os.path.join(data_dir, a, b, 'SUV.nii.gz'))
        Val_label = sitk.ReadImage(os.path.join(data_dir, a, b, 'SEG.nii.gz'))
        val_outputs = resample_3d(val_outputs, image_pt.GetSize())
        print(val_outputs.shape)
        nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(valpath, img_name))
        sitk.WriteImage(image_pt, os.path.join(valpath_image, img_name + '.nii.gz'))
        sitk.WriteImage(Val_label, os.path.join(valpath_gt, img_name + '_SEG.nii.gz'))
#
print("finished validation!")

# #
print("starting Test set inference")
test_org_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct", "label"]),
        # AddChanneld(keys=["image_pt", "image_ct", "label"]),
        EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
        ScaleIntensityRanged(keys=["image_ct"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
        ScaleIntensityRanged(keys=["image_pt"], a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
        Spacingd(keys=["image_pt", "image_ct", "label"], pixdim=(2, 2, 3), mode=("bilinear", "bilinear", "nearest")),
        # SpatialPadd(keys=["image_pt", "image_ct", "label"], spatial_size=window_size),
        # EnsureTyped(keys=["image_pt","image_ct", "label"],device=device, track_meta=False),
        # RandCropByLabelClassesd(keys=["image_pt", "image_ct", "label"], label_key="label", spatial_size=window_size, ratios=[2, 1],num_classes=2, num_samples=6),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),  # concatenate pet and ct channels
        EnsureTyped(keys=["image_petct", "label"]),
])

#
test_org_ds =  CacheDataset(data=Test_dicts, transform=test_org_transforms, cache_rate=0.1, num_workers=2, copy_cache=False)
test_org_loader = ThreadDataLoader(test_org_ds, batch_size=1, num_workers=0)
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

with torch.no_grad():
    for i, test_data in enumerate(test_org_loader):
        test_count += 1
        test_inputs, test_labels = (test_data["image_petct"].to(device), test_data["label"].to(device))
        original_affine = test_data["label_meta_dict"]["affine"][0].numpy()
        # print(original_affine)
        # print(test_labels.shape)
        _, _, h, w, d = test_labels.shape
        target_shape = (h, w, d)
        a, b = test_data["image_petct"].meta["filename_or_obj"][0].split("/")[7], \
        test_data["image_petct"].meta["filename_or_obj"][0].split("/")[8]

        img_name = a + "_" + b[:10]
        # print(test_data["image_petct"].meta["filename_or_obj"][0].split("/")[6],
        #       test_data["image_petct"].meta["filename_or_obj"][0].split("/")[7], img_name)
        print("Inference on case {}".format(img_name))
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_inputs, window_size, sw_batch_size, model,
                                               mode="gaussian", device=device)
        test_outputs = torch.softmax(test_outputs, 1).cpu().numpy()
        test_outputs = np.argmax(test_outputs, axis=1).astype(np.uint8)[0]
        test_labels = test_labels.cpu().numpy()[0, 0, :, :, :]
        image_pt = sitk.ReadImage(os.path.join(data_dir, a, b, 'SUV.nii.gz'))
        test_label = sitk.ReadImage(os.path.join(data_dir, a, b, 'SEG.nii.gz'))
        test_outputs = resample_3d(test_outputs, image_pt.GetSize())
        nib.save(nib.Nifti1Image(test_outputs.astype(np.uint8), original_affine), os.path.join(testpath, img_name))
        sitk.WriteImage(image_pt, os.path.join(testpath_image, img_name + '.nii.gz'))
        sitk.WriteImage(test_label, os.path.join(testpath_gt, img_name + '_SEG.nii.gz'))
print("finished test!")