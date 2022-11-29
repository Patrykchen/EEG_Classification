# 基于EEG的运动想象分类

- 依赖包安装：可以通过代码文件中的`requirements.txt`文件安装运行工程的必要依赖包
- 代码文件
  - `boost.py`：XGBoost模型的搭建及处理，包括结果输出
  - `lstm.py`：LSTM模型的训练及模型保存（储存在`./ckpt/`目录下）
  - `eeginception.py`：EEG-Inception模型的结构及前向传播定义
  - `train_eeg.py`：EEG-Inception模型的训练和模型保存
  - `test.py`：XGBoost和LSTM模型的测试以及集成学习测试
  - `result.py`：EEG-Inception模型的测试以及最后的csv文件输出
- 模型文件
  - `eeg_final.pth`为训练后的EEG-Inception模型

