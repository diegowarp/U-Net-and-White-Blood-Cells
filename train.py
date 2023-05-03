from google.colab import drive
drive.mount('/content/drive/', force_remount = True)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, utils 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

!cp /content/drive/MyDrive/Colab\ Notebooks/testeNovoDataset/unet.py /content
import unet
from unet import UNET


!cp /content/drive/MyDrive/Colab\ Notebooks/testeNovoDataset/utils.py /content
import utils
from utils3v import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 575  # 1280 originally
IMAGE_WIDTH = 575  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/dataset/train_images/"
TRAIN_MASK_DIR = "/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/dataset/train_masks/"
VAL_IMG_DIR = "/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/dataset/val_images/"
VAL_MASK_DIR = "/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/dataset/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # To 1 class output use unsqueeze
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = torch.permute(targets, (0,3,1,2)).to(device=DEVICE)


        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Perspective(),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=2).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

    # print some examples to a folder
    save_predictions_as_imgs(
        val_loader, model, folder="/content/drive/MyDrive/Colab Notebooks/testeNovoDataset/saved_images", device=DEVICE
    )


if __name__ == "__main__":
    main()