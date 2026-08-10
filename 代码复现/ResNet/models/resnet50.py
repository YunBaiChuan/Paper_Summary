import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU()
        
        if strides != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_channels = 64

        # 用 Bottleneck 替代 BasicBlock，且 Bottleneck 输出是 out_channels * 4
        self.stage1 = self._make_stage(64, 3, 1)
        self.stage2 = self._make_stage(128, 4, 2)
        self.stage3 = self._make_stage(256, 6, 2)
        self.stage4 = self._make_stage(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.relu = nn.ReLU()

        # 5次下采样: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.fc = nn.Linear(512 * 4 * 1 * 1, 10)

    def _make_stage(self, out_channels, blocks, strides):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(Bottleneck(self.in_channels, out_channels, strides))
            else:
                layers.append(Bottleneck(self.in_channels, out_channels, 1))

            self.in_channels = out_channels * 4

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