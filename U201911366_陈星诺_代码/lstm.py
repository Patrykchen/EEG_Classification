import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from boost import X_train,Y_train,X_test,Y_test # 从另一个文件中进行数据预处理

import numpy as np
import scipy.io as scio
import os

from tensorboardX import SummaryWriter

import random
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from datetime import datetime

traindata_dir = './data/train/'
testdata_dir = './data/test/'
checkpoints_dir = './ckpt/'

# 训练过程需要的超参数
lr = 0.001
epochs = 100
batch_size = 16
loss_fn = torch.nn.MSELoss()    # 选用均方误差作为损失函数
device = torch.device('cuda')


# 模型构建
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=9750,    # 将原始样本特征展平
            hidden_size=4096,   # 隐层节点个数
            batch_first=True,
            num_layers=1,
        )

        self.out = nn.Linear(4096, 1)   # 全连接层

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out)
        return out

# 将训练样本变为tensor格式，便于pytorch训练
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
data_loader_test = DataLoader(test_data, 1, sampler=sampler_test,drop_last=False)   # 用Dataloader类封装，便于批量训练

if __name__ == '__main__':

    writer = SummaryWriter('./runs')    # 保存结果到指定的目录
    sampler_train = torch.utils.data.RandomSampler(train_data)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler_train)

    model = LSTM()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay = 5)    # 使用Adam算法
    loss_epoch = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        n = 0

        for data in train_loader:
            n = n + 1
            input = data['X']
            label = data['y']
            input = input.to(torch.float32)
            label = label.to(torch.float32)

            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = loss_fn(label, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss_epoch.append(running_loss / n)
        writer.add_scalar('Train/Loss', running_loss / n, epoch)

        # 模型保存
        print("{} epoch loss:{:.5f}".format(epoch, loss_epoch[epoch]))
        if epoch % 50 == 0:
            filename_tail = datetime.strftime(datetime.now(), '%H_%M')
            checkpoint_latest_path = 'latest_' + filename_tail + '.pth'
            checkpoint_latest_path = os.path.join(checkpoints_dir, checkpoint_latest_path)
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, checkpoint_latest_path)
