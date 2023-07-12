# Facial-Expression-Recognition

A simple facial expression recognition system based on Python.

### 大概的思路

大概流程为：`视频切片`、`图像预处理（高斯模糊等）`、`人脸识别`、`分割人脸`、`使用训练好的表情模型`、`打上表情标签`

前面几步基于`opencv`的人脸库来捕捉，后面训练模型用`pytorch`对给定的灰度图来训练。
