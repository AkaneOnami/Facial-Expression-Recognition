import cv2
import numpy as np
import pandas as pd

file_path='./datasets/icml_face_data.csv'
split=24000
limit=28709

def CreateCsv():
    file_path='./datasets/icml_face_data.csv'
    docu=pd.read_csv(file_path)
    docu=docu.iloc[:,0]
    train=list()
    val=list()
    train.append(['document','emotion'])
    val.append(['document','emotion'])
    for u in range(len(docu)):
        if u < split:
            train.append([str(u)+'.jpg', str(docu[u])])
        elif u<limit:
            val.append([str(u)+'.jpg', str(docu[u])])
    pd.DataFrame(train).to_csv('./datasets/train_datasets.csv', index=False,header=False)
    pd.DataFrame(val).to_csv('./datasets/val_datasets.csv', index=False,header=False)
    
if __name__ == '__main__':
    CreateCsv()