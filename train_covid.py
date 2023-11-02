import os
import json
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from data import CovidDataset
from utils import data_split
import models
import losses
from metrics import SegmentationMetrics

exp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
exp_name = 'covid_mos_med'
model_name = 'att_unet'
loss_name = 'bce_dice_loss'
exp_dir = os.path.join('./experiments', exp_name, exp_time)
os.makedirs(exp_dir, exist_ok=True)

# hyperparameters
data_root = os.path.join('/data2/siyuan/covid19/processed_data/', exp_name)
batch_size = 10
lr = 1e-4
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'using device: {device}')

# data
train_datas, val_datas = data_split(data_root, exp_name, 0.8)
print(f'Using {exp_name} data: trained under {len(train_datas["imgs"])} datas and validated under {len(val_datas["imgs"])} datas!')
# print(f'train_img_paths: {train_datas["imgs"]}')
# print(f'train_mask_paths: {train_datas["masks"]}')
# print(f'val_img_paths: {val_datas["imgs"]}')
# print(f'val_mask_paths: {val_datas["masks"]}')

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_set = CovidDataset(datas=train_datas,
                         transform=train_transform)
val_set = CovidDataset(datas=val_datas,
                          transform=val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# model
model = models.__dict__[model_name](in_channels=1, out_channels=1).to(device)
# summary(model, (1, 256, 256))
print(f'Using {model_name} model!')
print(f'model parameters: {sum(p.numel() for p in model.parameters())}')

# loss and metrics
criterion = losses.__dict__[loss_name]()
print(f'Using {loss_name} loss!')

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

# train
train_loss_dict = {'bce_loss': {},
                   'dice_loss': {},
                   'total_loss': {}}
val_loss_dict = {'bce_loss': {},
                 'dice_loss': {},
                 'total_loss': {}}
best_train_loss = np.inf
best_val_loss = np.inf

train_metrics_dict = {'f1': {},
                      'asd': {},
                      'hd95': {},
                      }
val_metrics_dict = {'f1': {},
                    'asd': {},
                    'hd95': {},
                    }

for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader)
    train_bce_loss = 0.0
    train_dice_loss = 0.0
    train_total_loss = 0.0
    train_f1 = 0.0
    # train_asd = 0.0
    # train_hd95 = 0.0
    for i, (img, mask, name) in enumerate(pbar):
        img = img.to(device)
        mask = mask.to(device)
        # idx = sample['idx']
        optimizer.zero_grad()
        pred = model(img)

        bce_loss, dice_loss = criterion(pred, mask)
        alpha = 0.5
        total_loss = alpha*bce_loss + (1-alpha)*dice_loss
        total_loss.backward()
        optimizer.step()

        score = SegmentationMetrics(pred, mask)
        train_f1 = score.f1_score()
        # train_asd = score.asd()
        # train_hd95 = score.hausdorff_95()

        train_bce_loss += bce_loss.item()
        train_dice_loss += dice_loss.item()
        train_total_loss += total_loss.item()
        pbar.set_description(f'Train_Epoch:{epoch+1}/{epochs}|iter:{i+1}| bce_loss: {bce_loss.item():.4f} | dice_loss: {dice_loss.item():.4f} | total_loss: {total_loss.item():.4f}')
    train_bce_loss /= len(train_loader)
    train_dice_loss /= len(train_loader)
    train_total_loss /= len(train_loader)
    train_loss_dict['bce_loss'][epoch+1] = train_bce_loss
    train_loss_dict['dice_loss'][epoch+1] = train_dice_loss
    train_loss_dict['total_loss'][epoch+1] = train_total_loss
    train_metrics_dict['f1'][epoch+1] = train_f1
    # train_metrics_dict['asd'][epoch] = train_asd
    # train_metrics_dict['hd95'][epoch] = train_hd95

    print(f'Train_Epoch:{epoch+1}/{epochs}|train_bce_loss: {train_bce_loss:.4f}|train_dice_loss: {train_dice_loss:.4f}|train_total_loss: {train_total_loss:.4f}|train_f1: {train_f1:.4f}')

    model.eval()
    pbar = tqdm(val_loader)
    val_bce_loss = 0.0
    val_dice_loss = 0.0
    val_total_loss = 0.0
    val_f1 = 0.0
    # val_asd = 0.0
    # val_hd95 = 0.0
    for i, (img, mask, name) in enumerate(pbar):
        img = img.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            pred = model(img)

        bce_loss, dice_loss = criterion(pred, mask)
        alpha = 0.5
        total_loss = alpha*bce_loss + (1-alpha)*dice_loss

        score = SegmentationMetrics(pred, mask)
        val_f1 = score.f1_score()
        # val_iou = score.asd()
        # val_hd95 = score.hausdorff_95()

        val_bce_loss += bce_loss.item()
        val_dice_loss += dice_loss.item()
        val_total_loss += total_loss.item()
        pbar.set_description(f'Val_Epoch:{epoch+1}/{epochs}|iter:{i+1}| bce_loss: {bce_loss.item():.4f} | dice_loss: {dice_loss.item():.4f} | total_loss: {total_loss.item():.4f}')
    val_bce_loss /= len(val_loader)
    val_dice_loss /= len(val_loader)
    val_total_loss /= len(val_loader)
    val_loss_dict['bce_loss'][epoch+1] = val_bce_loss
    val_loss_dict['dice_loss'][epoch+1] = val_dice_loss
    val_loss_dict['total_loss'][epoch+1] = val_total_loss
    val_metrics_dict['f1'][epoch+1] = val_f1
    # val_metrics_dict['asd'][epoch] = val_asd
    # val_metrics_dict['hd95'][epoch] = val_hd95

    print(f'Val_Epoch:{epoch+1}/{epochs}|val_bce_loss: {val_bce_loss:.4f}|val_dice_loss: {val_dice_loss:.4f}| val_total_loss: {val_total_loss:.4f}|val_f1: {val_f1:.4f}')
    
    scheduler.step(val_total_loss)
    if val_total_loss < best_val_loss:
        best_epoch = epoch+1
        best_val_loss = val_total_loss
        torch.save(model.state_dict(), os.path.join(exp_dir, 'best_val_'+ str(epoch+1) + '.pth'))
        print(f'Best val loss: {best_val_loss:.4f} is saved!')
    if train_total_loss < best_train_loss:
        best_train_loss = train_total_loss
        torch.save(model.state_dict(), os.path.join(exp_dir, 'best_train_'+ str(epoch+1) + '.pth'))
        print(f'Best train loss: {best_train_loss:.4f} is saved!')
torch.save(model.state_dict(), os.path.join(exp_dir, 'last_'+ str(epochs) + '.pth'))
print(f'Last model is saved!')

# save loss and metrics
with open(os.path.join(exp_dir, 'train_loss.json'), 'w') as f:
    json.dump(train_loss_dict, f)
with open(os.path.join(exp_dir, 'val_loss.json'), 'w') as f:
    json.dump(val_loss_dict, f)
with open(os.path.join(exp_dir, 'train_metrics.json'), 'w') as f:
    json.dump(train_metrics_dict, f)
with open(os.path.join(exp_dir, 'val_metrics.json'), 'w') as f:
    json.dump(val_metrics_dict, f)
print(f'Loss and metrics are saved!')

# load the best model and save the predictions
model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_val_' + str(best_epoch) + '.pth')))
model.eval()
pbar = tqdm(val_loader)
for i, (img, mask, name) in enumerate(pbar):
    img = img.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        pred = model(img)
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for j in range(pred.shape[0]):
        pred_root = os.path.join(exp_dir, 'pred')
        mask_root = os.path.join(exp_dir, 'mask')
        os.makedirs(pred_root, exist_ok=True)
        os.makedirs(mask_root, exist_ok=True)
        pred_path = os.path.join(pred_root, f'{name[j]}_pred.npy')
        mask_path = os.path.join(mask_root, f'{name[j]}_mask.npy')
        np.save(pred_path, pred[j])
        np.save(mask_path, mask[j])
print(f'Predictions are saved!')
