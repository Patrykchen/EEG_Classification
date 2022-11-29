import torch
import tqdm
from eeginception import EEGInception
from torch.utils.data import DataLoader
from torch import nn, optim
import h5py
import numpy as np
import scipy.io as scio
from enum import Enum
from torch.utils.data import Dataset
import os
from tensorboardX import SummaryWriter

from scipy import signal

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda')

# 利用八阶巴特沃夫滤波器进行滤波
def data_aug(data):
    b, a = signal.butter(8, [0.125, 0.875], btype='band')
    buffer_x_test = signal.filtfilt(b[:4], a[:4], data, axis=0)
    noise = data - buffer_x_test
    return buffer_x_test, noise


traindata_dir = './data/train/'
testdata_dir = './data/test/'
batch_size = 8

X_train = []
Y_train = []
X_test = []
Y_test = []
file_list = os.listdir(traindata_dir)
for file in file_list:
    if os.path.splitext(file)[1] == '.mat': # 读取训练集下的数据文件
        data = scio.loadmat(traindata_dir + file)
        flag = True
        for x, y in zip(data['X'], data['y'].transpose()):
            x0, y0 = x, y
            '''# 对数据进行数据增强
            X_train.append(x.reshape(1, x.shape[0], x.shape[1]))
            if y[:] == 0:
                y = np.array([1,0])
            else:
                y = np.array([0,1])
            Y_train.append(y)
            if flag:
                flag = False
                x, const_noise = data_aug(x)
            x0, noise = data_aug(x0)
            x0 += const_noise'''
            X_train.append(x0.reshape(1,x0.shape[0],x0.shape[1]))
            if y0[:] == 0:
                y0 = np.array([1,0])
            else:
                y0 = np.array([0,1])
            Y_train.append(y0)

# 读取测试集下的数据文件
file_list = os.listdir(testdata_dir)
for file in file_list:
    if os.path.splitext(file)[1] == '.mat':
        data = scio.loadmat(testdata_dir + file)
        for x in data['X']:
            X_test.append(x.reshape(1,x.shape[0],x.shape[1]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# 将数据转换成tensor的数据格式
train_data = []
test_data = []
for x, y in zip(X_train, Y_train):
    data = {}
    data['X'] = torch.from_numpy(x)
    data['y'] = torch.from_numpy(y)
    train_data.append(data)

for x, y in zip(X_test, Y_test):
    data = {}
    data['X'] = torch.from_numpy(x)
    data['y'] = torch.from_numpy(y)
    test_data.append(data)

train_data = np.array(train_data)
test_data = np.array(test_data)

sampler_test = torch.utils.data.SequentialSampler(test_data)
val_loader = DataLoader(test_data, 1, sampler=sampler_test,drop_last=False)

sampler_train = torch.utils.data.RandomSampler(train_data)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
train_loader = DataLoader(train_data, batch_sampler=batch_sampler_train)

writer = SummaryWriter('./final')   # 输出TensorBoard结果到指定目录

if __name__ == '__main__':
    model = EEGInception(2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)

    with tqdm.trange(100) as t:
        loss_list = []
        for epoch in t:
            for data in train_loader:
                feat, label = data['X'], data['y']
                feat = feat.to(torch.float32)
                label = label.to(torch.float32)
                feat = feat.to(device)
                label = label.to(device)
                out = model(feat)
                loss = loss_fn(out, label)
                loss_list.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=loss.item())
                val_mean_loss = sum(loss_list) / len(loss_list)
            writer.add_scalar('Train/Loss', val_mean_loss, epoch)

        torch.save(model.state_dict(), f"./eeg.pth")   # 模型保存