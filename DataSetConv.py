import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
from models.ModelDefinition import *

class CustomDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.data = cv2.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    
    train_folder = "./datasets/cnn_train/"
    train_csv = "./datasets/train_datasets.csv"

    val_folder = "./datasets/cnn_val/"
    val_csv = "./datasets/val_datasets.csv"

    # 创建训练集数据集对象
    train_dataset = CustomDataset(train_folder, train_csv, transform=transform)

    # 创建验证集数据集对象
    val_dataset = CustomDataset(val_folder, val_csv, transform=transform)

    # 保存数据集对象
    torch.save(train_dataset, './customDatasets/train_dataset.pt')
    torch.save(val_dataset, './customDatasets/val_dataset.pt')

if __name__ =='__main__':
    main()
