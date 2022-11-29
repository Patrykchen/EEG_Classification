'''
用来利用EEG模型输出最后的CSV文件
'''

from eeginception import EEGInception
import torch
from train_eeg import val_loader
import numpy as np
from train_eeg import testdata_dir
import os
import scipy.io as scio
from torch.utils.data import DataLoader
import pandas as pd

X_test = []
dic = {}    # 用字典来存储数据
file_list = os.listdir(testdata_dir)
for file in file_list:
    if os.path.splitext(file)[1] == '.mat':
        dic[os.path.splitext(file)[0]] = []
        data = scio.loadmat(testdata_dir + file)
        for x in data['X']:
            dic[os.path.splitext(file)[0]].append(x.reshape(1,x.shape[0],x.shape[1]))

model_path = './eeg1.pth'
device = torch.device('cuda')

model = EEGInception(2)
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt)
model.to(device)

cnt, acc = 0, 0

results = {}    # 用字典存储结果便于转换成DateFrame格式最后输出到csv文件中

with torch.no_grad():
    for key in dic:
        results[key] = []
        for data in dic[key]:
            feat = torch.FloatTensor(data)
            feat = feat.to(torch.float32)
            feat = torch.unsqueeze(feat,0)

            feat = feat.to(device)

            out = model(feat)

            out = 0 if out[0,0] > out[0,1] else 1
            results[key].append(out)
        results[key] = np.array(results[key])

df=pd.DataFrame(results)
df.to_csv('result.csv',index=False)

print(results)
