import torch.nn as nn 

class Plain34(nn.Module):
    def __init__(self):
        super().__init__()

        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_channels = 64

        # 堆叠层
        self.stage1 = self._make_stage(64, 3, 1)
        self.stage2 = self._make_stage(128, 4, 2)
        self.stage3 = self._make_stage(256, 6, 2)
        self.stage4 = self._make_stage(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.relu = nn.ReLU()

        # 5次下采样: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.fc = nn.Linear(512 * 1 * 1, 10)

    def _make_stage(self, out_channels, blocks, strides):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1, stride=strides))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x