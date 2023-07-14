import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import (
    Grayscale,
    TenCrop,
    Lambda,
    Normalize,
    RandomResizedCrop,
    ColorJitter,
    RandomAffine,
    RandomHorizontalFlip,
    RandomRotation,
    FiveCrop,
    RandomErasing
)


def get_transform(arugment=True):
    if arugment==True:
        transform = transforms.Compose([
            transforms.Resize((48,48)),

            Grayscale(),
            RandomResizedCrop((48,48)),
            ColorJitter(),
            RandomAffine(degrees=0, translate=(0.1, 0.1)),
            RandomHorizontalFlip(),
            RandomRotation(degrees=10),
            FiveCrop(size=40),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            RandomErasing(),
            
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((48,48)),
            Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return transform

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.3)
        self.dropout3=nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        
        # 换用numpy以应用cv2提供的库
        image = np.array(image)
        # 高斯去噪
        image = cv2.GaussianBlur(image, (7, 7), 5)
        # 中值滤波
        image = cv2.medianBlur(image, 5)
        # 换回来
        image = Image.fromarray(image)
        
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
 
        '''
        如果步长 stride 不为 1 或者输入通道数 in_planes 不等于扩展系数 self.expansion 乘以输出通道数 planes，
        则需要进行维度匹配，使用一个包含一个卷积层和一个批量归一化层的序列 self.shortcut 来进行匹配
        '''
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 

 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

transform=get_transform()