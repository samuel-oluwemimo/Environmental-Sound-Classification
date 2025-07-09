import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride !=1 or in_channel != out_channel
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out += shortcut
        out = nn.ReLU()(out)

        return out

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))

        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128, 128) for i in range(4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256, 256) for i in range(6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512, 512) for i in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x