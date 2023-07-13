# 说明

该文件下的数据集的文件结构为：

1. `./datasets/cnn_train/` 和 `./datasets/cnn_val/` ： 参考kaggle上的
[原数据集](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv)
给定的像素生成，按照 `0~23999` 和 `24000~28708` 将原数据集分为 `8:2` 的比例
2. `./datasets/train_datasets.csv` 和 `./datasets/val_datasets.csv` ： 参照给出的 `./datasets/icml_face_data` 进行的划分。