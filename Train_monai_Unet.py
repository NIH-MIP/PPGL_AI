import os
import torch
import time

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.transforms.utils import ndimage
from monai.utils import first
from tqdm import tqdm

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
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
)



from datasetcheck import get_data, crop_get_data
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("using cuda:0")
else:
    raise RuntimeError("this tutorial is intended for GPU, but no CUDA device is available")

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))



'''Set Learning rate'''
lr = 1e-2
'''Set number of epochs'''
numephs = 3000

'''Set Model name'''
modelname = 'UNet_whole_crops' 
# Provided the path where output directory should be
main_dir='path-to-output-directory'
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

#Provided the path where the dataset folder is
data_dir = 'Path-to-dataset-folder'

#setup output folders
model_path= os.path.join(main_dir,'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)
out_path= os.path.join(main_dir, 'results')
if not os.path.exists(out_path):
    os.makedirs(out_path)

'''New way to split data stratified way.'''

Train_dicts, Test_dicts, Val_dicts= get_data (data_dir)

train_transforms = Compose([
            LoadImaged(keys=["image_pt","image_ct", "label"]),
            EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
            ScaleIntensityRanged(keys=["image_ct"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
            ScaleIntensityRanged(keys=["image_pt"], a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
            Spacingd(keys=["image_pt","image_ct", "label"], pixdim=(2.0364201068878174, 2.0364201068878174, 3.0), mode=("bilinear", "bilinear", "nearest")),
            # SpatialPadd(keys=["image_pt","image_ct", "label"], spatial_size=window_size),
            # RandCropByLabelClassesd(keys=["image_pt", "image_ct", "label"], label_key="label", spatial_size=window_size, ratios=[2, 1],num_classes=2, num_samples=6),
            ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct",dim=0),  # concatenate pet and ct channels
            EnsureTyped(keys=["image_petct", "label"]),
])

val_transforms = Compose([
        LoadImaged(keys=["image_pt", "image_ct", "label"]),
        EnsureChannelFirstd(keys=["image_pt", "image_ct", "label"]),
        ScaleIntensityRanged(keys=["image_ct"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
        ScaleIntensityRanged(keys=["image_pt"], a_min=0, a_max=15,b_min=0.0, b_max=1.0, clip=False),
        Spacingd(keys=["image_pt","image_ct", "label"], pixdim=(2, 2, 3), mode=("bilinear", "bilinear", "nearest")),
        # SpatialPadd(keys=["image_pt","image_ct", "label"], spatial_size=window_size),
        # RandCropByLabelClassesd(keys=["image_pt", "image_ct", "label"], label_key="label", spatial_size=window_size, ratios=[2, 1],num_classes=2, num_samples=6),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),  # concatenate pet and ct channels
        EnsureTyped(keys=["image_petct", "label"]),
])

print("hello!")

# create a training data loader


check_train = CacheDataset(data=Train_dicts, transform= train_transforms, cache_rate=0.05, num_workers=4,copy_cache=False)
train_loader = ThreadDataLoader(check_train, batch_size=2, shuffle=True, num_workers=0)


check_val = CacheDataset(data=Val_dicts, transform= val_transforms, cache_rate=0.01, num_workers=2, copy_cache=False)
val_loader = ThreadDataLoader(check_val, batch_size=1, num_workers=0, pin_memory=True)



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

# start a typical PyTorch training
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
epoch_times = []
#resume training using checkpoint
# epoch= 26
path= os.path.join(model_path, "UNet_whole_crops_best_epoch-195.pth")
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
last= os.path.join(model_path, "UNet_whole_cropsModel_last_epoch.pth")
check= torch.load(last)
epoch_loaded = check['epoch']
print("Loaded model Epoch, ", epoch_loaded)
loss = checkpoint['loss']
print(model)
# and training loop should be in range (epoch, numephs)

#set the last epoch
# start_epoch = 0
# if start_epoch > 0:
#     resume_epoch = start_epoch - 1
#     resume(model,os.path.join(model_path, modelname+f"epoch-{resume_epoch}.pth"))

for epoch in range(epoch_loaded,numephs):
    print(epoch)
    print(f"epoch {epoch}/{numephs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (batch_data["image_petct"].to(device), batch_data["label"].to(device))
        # print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(check_train) // train_loader.batch_size
    print(epoch_loss)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
    epoch_train=time.time()
    print(f"training for this epoch finished")
    # check_train.update_cache()
    if (epoch + 1) % val_interval == 0:
        metric_values = []
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["image_petct"].to(device), val_data["label"].to(device)
                roi_size = [128,128,32]
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model, overlap=0.6, sw_device=device, device=device)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            f_path=  os.path.join(model_path, modelname+f"Model_last_{epoch}.pth")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_function,
                        'dice': metric},
                       f_path)
            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save({'epoch': epoch,
                            'model_state_dict':  model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss':loss_function,
                            'dice': metric},
                            os.path.join(model_path, modelname+f"_best_epoch-{epoch}.pth"))
                # checkpoint(model, os.path.join(model_path, modelname+f"epoch-{epoch}.pth"))
                print("saved new best metric model")

            # Logger bar
            print(
                "current epoch: {} Validation dice: {:.4f} Best Validation dice: {:.4f} at epoch {}".format(
                    epoch, metric, best_metric, best_metric_epoch
                )
            )
            epoch_val_e = time.time()
            print(f"validation for this epoch finished")
print(f"train completed, best_metric: {best_metric:.4f}", f" at epoch: {best_metric_epoch}")