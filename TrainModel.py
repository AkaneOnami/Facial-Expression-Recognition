import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.ModelDefinition import *

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
train_path = './customDatasets/train_dataset.pt'
val_path = './customDatasets/val_dataset.pt'

# 设置训练参数
batch_size = 120
learning_rate = 0.0008
num_epochs = 30

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

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # 保存模型
    model = model.to('cpu')
    torch.save(model.state_dict(), './models/cnn_model.pth')
    
    # 验证用
    model=CNN()
    model.load_state_dict(torch.load('./models/cnn_model.pth')) 
    model.to(device) 
    
    # 验证模型
    model.eval()
    
    vaild_loss = []
    vaild_ac = []
    y_pred = []
    
    # 正确的数量
    correct=0;
    
    with torch.no_grad():
        val_dataset = torch.load(val_path)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for i, (images, labels) in enumerate(val_loader):
            # 将输入数据和标签移动到设备
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            pred = outputs.max(1,keepdim=True)[1]
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    vaild_ac.append(correct/len(val_dataset)) 
    vaild_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss,correct,len(val_dataset),100.*correct/len(val_dataset)))
    
    # 画热力图
    emotion = ["angry","disgust","fear","happy","sad","surprised","neutral"]
    sns.set()
    f,ax=plt.subplots()
    y_true = [emotion[i] for _,i in val_dataset]
    y_pred = [emotion[i] for i in y_pred]
    C2= confusion_matrix(y_true, y_pred, labels=["angry","disgust","fear","happy","sad","surprised","neutral"])#[0, 1, 2,3,4,5,6])

    sns.heatmap(C2,annot=True ,fmt='.20g',ax=ax) #热力图

    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.savefig('./images/matrix.jpg')
        
if __name__ == '__main__':
    main()