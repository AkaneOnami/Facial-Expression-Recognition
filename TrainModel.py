import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.ModelDefinition import *

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = './custom_datasets.pt'

# 设置训练参数
batch_size = 100
learning_rate = 0.001
num_epochs = 10

def main():
    
    # # 加载训练数据集
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, # transfrom定义在模型中作为转换函数
    #                                            download=True)
    train_dataset = torch.load(train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 初始化模型和损失函数
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 将输入数据和标签移动到设备
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # 保存模型
    model = model.to('cpu')
    torch.save(model.state_dict(), 'cnn_model.pth')


if __name__ == '__main__':
    main()