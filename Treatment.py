import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from models.ModelDefinition import *

def getSliceAndDraw(url='./videos/sample.mp4'):
    global arugment
    
    arugment=False
    # 拉取视频
    video=cv2.VideoCapture(url)
    
    # opencv自带的人脸分类器
    face_path='./xmls/haarcascade_frontalface_default.xml'
    face_cas=cv2.CascadeClassifier(face_path)
    face_cas.load(face_path)
    
    # 导入先前训练的模型
    model=ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load('./models/cnn_model.pth'))
    model.eval()  # 将模型设置为评估模式
    
    emoji0=cv2.imread('./emoji/0-Angry.png')
    emoji1=cv2.imread('./emoji/1-Disgust.png')
    emoji2=cv2.imread('./emoji/2-Fear.png')
    emoji3=cv2.imread('./emoji/3-Happy.png')
    emoji4=cv2.imread('./emoji/4-Sad.png')
    emoji5=cv2.imread('./emoji/5-Surprise.png')
    emoji6=cv2.imread('./emoji/6-Neutral.png')
    
    while True:
        # 读取视频帧
        ret,frame=video.read()
        
        # 视频结束时退出
        if not ret:
            break
        
        # 转换成单通道灰度值
        cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        '''
        rect = classifier.detectMultiScale(gray, scaleFactor, minNeighbors, minSize, maxsize)
        参数:
        Gray:要进行检测的人脸图像
        scaleFactor:前后两次扫描中，搜索窗口的比例系数
        minneighbors:目标至少被检测到minNeighbors次才会被认为是目标
        minsize和maxsize:目标的最小尺寸和最大尺寸
        '''
        # 进行人脸和眼睛检测
        faceRects = face_cas.detectMultiScale(
            cvt_img, scaleFactor=1.2, minNeighbors=4, minSize=(15, 15))
        
        if len(faceRects) == 0:  # 如果获取失败
            # 只显示原始图像
            cv2.imshow('video',frame)
            if cv2.waitKey(20) & 0xFF==27:
                break
            continue  # 结束本次循环
        
        for faceRect in faceRects:
            x, y, w, h = faceRect
            # 框出人脸
            img_addROI = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # 截取人脸图像用于识别表情
            face=frame[y:y+h, x:x+w]
            
            # # 高斯去噪
            # blur = cv2.GaussianBlur(face, (7, 7), 5)
            # # 中值滤波
            # dst = cv2.medianBlur(blur, 5)
            # # 腐蚀, 去掉图中的小方块，腐蚀后的图像干净许多, 但物体变小
            # erode = cv2.erode(dst, kernel)
            # # 膨胀, 还原使物体变大, 膨胀3次大一点
            # dilate = cv2.dilate(erode, kernel, iterations=3)
            
            # 图像按照归一化 48*48的灰度图
            face=cv2.resize(face,(48,48))
            face=Image.fromarray(face)
            my_transform=get_transform(False)
            face=my_transform(face)
            face=torch.unsqueeze(face,dim=0) # 增加一维作为批次维度
            
            # 不启动梯度，即不使用该数据进行训练
            with torch.no_grad():
                output = model(face)
                # 得到训练后的表情标签
                predicted_label = torch.argmax(output, 1).item()

            # 利用得到的标签将对应的emoji打回到目录中
            if predicted_label==0:
                emoji=emoji0
            elif predicted_label==1:
                emoji=emoji1
            elif predicted_label==2:
                emoji=emoji2
            elif predicted_label==3:
                emoji=emoji3
            elif predicted_label==4:
                emoji=emoji4
            elif predicted_label==5:
                emoji=emoji5
            elif predicted_label==6:
                emoji=emoji6
            else:
                emoji=emoji6
                
            confidence = torch.max(torch.softmax(output, dim=1)).item()
            if confidence<0.70:
                emoji=emoji6
            
            if emoji is None:
                continue
            
            # 让emoji不要太大
            emoji=cv2.resize(emoji,(60,60))
            
            # 获取 emoji 图像的宽度和高度
            emoji_height, emoji_width, _ = emoji.shape     
            
            # 默认在人脸的左边显示
            frame[y:y+emoji_height,x-100:x+emoji_width-100,:]=emoji[:,:,:]
            
            # 显示图像
            cv2.imshow('video',frame)
            if cv2.waitKey(42) & 0xFF==27:
                break
 
    video.release()      

if __name__ == '__main__':
    getSliceAndDraw()
            
            
            
            
            
        
            
            

    