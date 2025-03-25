import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from pytorch_lightning import LightningModule, Trainer
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping


def read_csv(filename): # from run.py
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res


def get_transform(train=False):
    if train:
        return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.GaussianBlur(p=0.5, blur_limit=3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2(transpose_mask=True)
    ])
    else:
        return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True)
    ])

class BirdsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = list(self.data.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = self.data[img_name]
        return image, label

def split_data(csv_file, val_size=0.2):
    train_data, val_data = train_test_split(list(csv_file.items()), test_size=val_size, stratify=list(csv_file.values()))
    return dict(train_data), dict(val_data)


class BirdClassifier(LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, trainable_layers=6, fast_train=False):
        super().__init__()
        self.save_hyperparameters()

        mobilenet = models.mobilenet_v2(pretrained=False) # True !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.features = mobilenet.features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.additional_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.BatchNorm1d(mobilenet.last_channel),
            nn.ReLU()
        )
        self.classifier = nn.Linear(mobilenet.last_channel, num_classes)

        for param in self.features[:-trainable_layers].parameters(): # морозим
            param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.additional_layers(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate, weight_decay=0.0005)

def train_classifier(train_gt, train_img_dir, fast_train=True):
    train_data, val_data = split_data(train_gt, val_size=0.1)

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False) # !!!!!!!!!!!!!!!!

    train_dataset = BirdsDataset(train_data, train_img_dir, transform=train_transform)
    val_dataset = BirdsDataset(val_data, train_img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    num_classes = len(set(train_gt.values())) # 50

    model = BirdClassifier(num_classes=num_classes, trainable_layers=5, fast_train=fast_train)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5)
    ]


    trainer = Trainer(
        max_epochs=1 if fast_train else 50,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(model, train_loader, val_loader)
    return model

def classify(model_path, test_img_dir):
    transform = get_transform(train=False)
    model = BirdClassifier(num_classes=50).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    predictions = {}
    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        image = transform(image=image)['image'].unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, pred = torch.max(outputs, 1)
            predictions[img_name] = pred.item()
    return predictions


def main():
    train_gt = "public_tests/00_test_img_input/train/gt.csv"
    train_img_dir = "public_tests/00_test_img_input/train/images"
    train_gt = read_csv(train_gt)

    model = train_classifier(train_gt, train_img_dir, fast_train=False)
    torch.save(model.state_dict(), "./birds_model.pt")


if __name__ == "__main__":
    main()
