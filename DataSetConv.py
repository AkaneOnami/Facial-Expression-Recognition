import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
import pandas as pd
from models.ModelDefinition import *

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
