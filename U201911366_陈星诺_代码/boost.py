import numpy as np
import scipy.io as scio
import os
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error as RMSE
import torch
import random
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.cluster import KMeans

traindata_dir = './data/train/'
testdata_dir = './data/test/'

# 随机数种子便于结果复现
seed = 39
torch.manual_seed(seed)
random.seed(seed)

'''数据读取以及预处理
把S2——S3读取成训练集，S1为测试集'''
X_train = []
Y_train = []
X_test = []
Y_test = []
file_list = os.listdir(traindata_dir)
for file in file_list:
    if os.path.splitext(file)[1] == '.mat' and os.path.splitext(file)[0] != 'S1':
        data = scio.loadmat(traindata_dir + file)
        for x, y in zip(data['X'], data['y'].transpose()):
            X_train.append(x.flatten()) # 将原来13*750的向量展平
            Y_train.append(y)
    elif os.path.splitext(file)[0] == 'S1' and os.path.splitext(file)[1] == '.mat':
        data = scio.loadmat(traindata_dir + file)
        for x, y in zip(data['X'], data['y'].transpose()):
            X_test.append(x.flatten())
            Y_test.append(y)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


'''
# 可以读取测试集的数据，测试集没有标签
file_list = os.listdir(testdata_dir)
for file in file_list:
    if os.path.splitext(file)[1] == '.mat':
        data = scio.loadmat(testdata_dir + file)
        for x in data['X']:
            X_test.append(x.flatten())
        # Y_test.append(data['y'].transpose())
X_test = np.array(X_test)
# Y_test = np.array(Y_test)'''

# 聚类数据清洗
model = KMeans(n_clusters=2,random_state=seed)
model.fit(X_train)
yhat = model.predict(X_train)
clusters = np.unique(yhat)
maxlist = []
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    a = X_train[row_ix]
    c = np.mean(a, axis=0)  # 计算每一类的中心点
    max = 0
    for i in a:
        dist = np.linalg.norm(i - c)
        max = dist if dist > max else max
    maxlist.append(max) # 计算该类到中心点的距离最大值

tmp_y = []
tmp_x = []
n = 0

for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    a = X_train[row_ix]
    c = np.mean(a, axis=0)
    max = maxlist[n]
    for index, vec in zip(row_ix[0], a):
        if np.linalg.norm(vec - c) <= 0.9 * max:
            tmp_x.append(vec)
            tmp_y.append(Y_train[index])
    n += 1

X_train = np.array(tmp_x)   # 得到清洗后的数据及标签
Y_train = np.array(tmp_y)

drop = torch.nn.Dropout(p = 0.05)   # 以0.05的概率将训练集中某几个维度的向量值置零
X_train = torch.from_numpy(X_train)
X_train = drop(X_train)
X_train = np.array(X_train)

# xgboost实现
acc = 0
cnt = 0
reg = XGBR(n_estimators=300, gamma=0.1, max_depth=4, reg_alpha=0.5, reg_lambda=1).fit(X_train, Y_train)  # 训练
Y_pred = reg.predict(X_test)

if __name__ == '__main__':
    # 测试
    mid = np.median(Y_pred)
    for gt, pre in zip(Y_test, Y_pred):
        cnt += 1
        if gt == int(pre + 0.5):
            acc += 1
    print(acc / cnt)
    print(RMSE(Y_test, reg.predict(X_test)))