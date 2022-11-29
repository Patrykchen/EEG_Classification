import torch
import torch.nn as nn
import torch.nn.functional



class DepthWiseConv2d(nn.Module):
    def __init__(self, channel, kernel_size, depth_multiplier, bias=False):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size, bias=bias, groups=channel)
            for _ in range(depth_multiplier)
        ])

    def forward(self, x):
        output = torch.cat([net(x) for net in self.nets], 1)
        return output


class EEGInception(nn.Module):
    def __init__(self, num_classes, fs=1282, num_channels=13, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True)):  # 通道数需要修改一下，Inception是多个卷积核并行作用，卷积核个数由filters_per_branch控制
        super().__init__()
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    1, filters_per_branch, (scales_sample, 1),
                    padding="same"
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                DepthWiseConv2d(8, (1, num_channels), 2),
                nn.BatchNorm2d(filters_per_branch * 2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])
        self.avg_pool1 = nn.AvgPool2d((4, 1))

        
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    len(scales_samples) * 2 * filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False,
                    padding="same"
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))
        
        self.output = nn.Sequential(
            nn.Conv2d(
                24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                bias=False, padding='same'
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((1, 1)),   # 当前尺寸太小，下采样无法得到有效输出
            nn.Dropout(dropout_rate),

            nn.Conv2d(
                12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                bias=False, padding='same'
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            nn.AvgPool2d((1, 1)),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Sequential(
            nn.Linear(4428, num_classes),
            nn.Softmax(1)
        )   # 分类，全连接，全连接层的数量需要根据当前张量大小确定

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], dim=1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], dim=1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, start_dim=1)
        return self.cls(x)