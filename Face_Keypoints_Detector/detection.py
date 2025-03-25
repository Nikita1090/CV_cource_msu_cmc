import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import torch.optim as optim
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, img_dir, gt, transform=None):
        self.img_dir = img_dir
        self.gt = gt
        self.transform = transform
        self.filenames = list(gt.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        keypoints = self.gt[img_name].astype('float32')

        if self.transform:
            augmented = self.transform(image=np.array(image), keypoints=keypoints.reshape(-1, 2))
            image = augmented['image']
            keypoints = augmented['keypoints'].flatten()

        return image, keypoints
    


def get_transform(train=False):
    if train:
        return A.Compose([
            A.Resize(100, 100),
            A.ShiftScaleRotate(rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ToFloat(255),
            A.Normalize(max_pixel_value=1),
            ToTensorV2(transpose_mask=True)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(100, 100),
            A.ToFloat(255),
            A.Normalize(max_pixel_value=1),
            ToTensorV2(transpose_mask=True)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


"""class FaceKeypointsModel(nn.Module):
    def __init__(self):
        super(FaceKeypointsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 28)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""


class FaceKeypointsModel(nn.Module):
    def __init__(self):
        super(FaceKeypointsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 28)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




def train_detector(gt, img_dir, fast_train=False):
    batch_size = 10
    dataset = MyDataset(img_dir, gt, transform=get_transform(train=True))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FaceKeypointsModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

    num_epochs = 2 if fast_train else 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, keypoints in train_loader:
            images, keypoints = images.float().to(device), keypoints.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if fast_train == False:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, keypoints in val_loader:
                    images, keypoints = images.float().to(device), keypoints.float().to(device)
                    outputs = model(images)
                    loss = criterion(outputs, keypoints)
                    val_loss += loss.item()

            print(f'Validation Loss: {val_loss/len(val_loader)}')
    if fast_train == False:
        torch.save(model.state_dict(), 'facepoints_model.pt')
    return model


def detect(model_path, img_dir):
    model = FaceKeypointsModel().to("cpu")
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device('cpu')))
    model.eval()

    transform = get_transform(train=False)
    filenames = os.listdir(img_dir)
    results = {}

    for filename in filenames:
        img_path = os.path.join(img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        image = np.array(image)
        image = transform(image=image)['image'].unsqueeze(0).float()

        with torch.no_grad():
            keypoints = model(image).squeeze().numpy()

        keypoints = keypoints.reshape(-1, 2)
        keypoints[:, 0] *= original_size[0] / 100
        keypoints[:, 1] *= original_size[1] / 100
        keypoints = keypoints.flatten()

        results[filename] = keypoints

    return results


def read_csv(filename): # from run.py
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def main():
    csv_path = "public_tests/00_test_img_input/train/gt.csv"
    image_folder = "public_tests/00_test_img_input/train/images"
    train_gt = read_csv(csv_path)

    model = train_detector(train_gt, image_folder, fast_train=False)

    model_filename = 'facepoints_model.pt'
    torch.save(model.state_dict(), model_filename)

if __name__ == '__main__':
    main()
