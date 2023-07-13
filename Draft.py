
# 仅用于测试.pt文件 #

import torch
from models.ModelDefinition import *

# 加载.pt文件
dataset = torch.load('./customDatasets/train_dataset.pt')


# 打印相关属性
print("Dataset length:", len(dataset))
print("Sample 0:", dataset[0])
