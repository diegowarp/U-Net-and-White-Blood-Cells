# -*- coding: utf-8 -*-
"""utils2V.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SkNRlQv4XIn4i6lN2MGSm2_dILDOW4fU
"""

import torch
import torchvision
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        mask[mask<50] = 0
        mask[(mask>50) & (mask<150)] = 100
        mask[mask>150] = 255
        mask[:,:,0][mask[:,:,2]>99] = 1
        mask[:,:,1][mask[:,:,2]>254] = 0
        mask[:,:,1][mask[:,:,1]==100] = 1
        mask = np.delete(mask, 2, 2)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

def save_checkpoint(state, filename="/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MyDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MyDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device=device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = torch.permute(y, (0,3,1,2)).to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {100*dice_score/len(loader):.2f}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/saved_images", device=device
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = torch.permute(y, (0,3,1,2)).to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            preds = torch.cat((255*preds,255*preds[:,0,:,:].unsqueeze(1)), dim=1)
            y = torch.cat((255*y,255*y[:,0,:,:].unsqueeze(1)), dim=1)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}/y_{idx}.png")
        

    model.train()