import numpy as np

from lstm import data_loader_test
from lstm import LSTM
import torch
from boost import reg
model_path = './ckpt/latest_07_53.pth'
from boost import X_test,Y_test,Y_pred
from eeginception import EEGInception
from train_eeg import val_loader

# 读取LSTM模型
device = torch.device('cuda')
model = LSTM()
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.to(device)

# 读取EEG-Inception模型
model_path = './eeg.pth'
eeg = EEGInception(2)
ckpt = torch.load(model_path, map_location='cpu')
eeg.load_state_dict(ckpt)
eeg.to(device)

acc = 0
cnt = 0

result = []
result0 = []

# LSTM模型在训练集上测试并输出结果
with torch.no_grad():
    for data in X_test:
        input = torch.from_numpy(data)
        input = torch.unsqueeze(input,0)
        input = input.to(torch.float32)
        input = input.to(device)

        output = model(input)
        output = torch.squeeze(output,0)
        output = output.cpu()
        output = output.numpy()
        result.append(output)

# EEG-Inception模型在训练集上测试并输出结果
with torch.no_grad():
    for data in val_loader:
        feat = data['X']
        feat = feat.to(torch.float32)
        feat = feat.to(device)
        out = eeg(feat)
        out = np.array([0]) if out[0, 0] > out[0, 1] else np.array([1])
        result0.append(out)

tmp = Y_pred    # 直接从boost文件中读取XGBoost的分类结果
tmp = np.where(tmp >= 0.5, 1 ,0)    # 四舍五入法输出标签

result = np.array(result)   # result变量是LSTM模型的结果
result = result.flatten()
max = np.max(result)
min = np.min(result)
result = (result - min) / (max - min)   # 归一化到[0,1]
mid = np.median(result)
result = np.where(result > mid, 1, 0)   # 按照中位数法判断结果

result0 = np.array(result0) # result0是EEG模型的结果
result0 = result0.flatten()
result0 = np.where(result0 >= 0.5, 1 ,0)    # 四舍五入法输出结果


output = 0.2 * result + 0.2 * tmp + 0.8 * result0   # 可以通过调整权重修改投票器的“话语权”
ones = 0
zeros = 0
mid = np.median(output)

for pre, gt in zip(output, Y_test):
    cnt += 1
    pre = 1 if pre > mid else 0
    if gt == pre:
        if gt == 1:
            ones+=1
        else:
            zeros+=1
        acc += 1

print(acc / cnt)
print(zeros,ones)