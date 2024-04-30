import os
import torch
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import glob
import numpy as np
import torchvision
from monai.transforms import (
    Compose, RandFlipd, Affined, Rand3DElastic, ToTensord, EnsureChannelFirstD, DivisiblePadd
)

from model import *
from dataset import *
from criteria import *

inChans = 4
input_shape = (4, 160, 240, 240)
seg_outChans = 3
activation = "relu"
normalization = "group_normalization"
VAE_enable = True
train_img_root = 'Task01_BrainTumour/imagesTr'
train_label_root = 'Task01_BrainTumour/labelsTr'
val_img_root = 'Task01_BrainTumour/imagesTr'
val_label_root = 'Task01_BrainTumour/labelsTr'
train_batch_size = 1
val_batch_size = 1
checkpoint_path = 'content'
epochs = 100
lr = 0.001

train_transforms = Compose([
    EnsureChannelFirstD(['label'],channel_dim='no_channel'),
    #AddChanneld(['label']),
    RandFlipd(['image', 'label'], prob=0.5, spatial_axis=0),
    RandFlipd(['image', 'label'], prob=0.5, spatial_axis=1),
    DivisiblePadd(k=8, keys=['image', 'label']),
    ToTensord(['image', 'label'])
])

val_transforms = Compose([
    EnsureChannelFirstD(['label'],channel_dim='no_channel'),
    DivisiblePadd(k=8, keys=['image', 'label']),
    ToTensord(['image', 'label'])
])

train_dataset = BraTSDataSet(img_root=train_img_root, label_root=train_label_root, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size) #num_workers=os.cpu_count())

val_dataset = BraTSDataSet(img_root=val_img_root, label_root=val_label_root, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

net = NvNet(inChans, input_shape, seg_outChans, activation, normalization, VAE_enable, mode='trilinear')
if torch.cuda.is_available(): net = net.cuda()

criterion = CombinedLoss(k1=0.1, k2=0.1)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

import math
best_loss = -math.inf


for epoch in range(0, epochs):
    # Train Model
    print('\n\n\nEpoch: {}\n'.format(epoch))
    net.train(True)
    loss = 0
    #lr = lr * (0.5 ** (epoch // 4))
    lr = lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    torch.set_grad_enabled(True)
    for idx, (img, label) in enumerate(train_loader):
        if torch.cuda.is_available():
          img, label = img.cuda(), label.cuda()
        pred = net(img)
        seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
        batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += float(batch_loss)
    log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
    print(log_msg)


    # Validate Model
    print('\n\n')
    net.eval()
    for module in net.module.modules():
        if isinstance(module, torch.nn.modules.Dropout2d):
            module.train(True)
        elif isinstance(module, torch.nn.modules.Dropout):
            module.train(True)
        else:
            pass
    loss = 0
    torch.set_grad_enabled(False)
    for idx, (img, label) in enumerate(val_loader):
      if torch.cuda.is_available():
        img, label = img.cuda(), label.cuda()
        pred = net(img)
        seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
        batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
        loss += float(batch_loss)
    log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
    print(log_msg)

    # Save Model
    if loss <= best_loss:
        torch.save(os.path.join(checkpoint_path, f'epoch:{epoch}_loss{loss}.tar'))
        best_loss = loss
        print("Saving...")
