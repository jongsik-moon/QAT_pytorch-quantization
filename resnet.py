import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
            self.downsample = True
        else:
            self.shortcut = nn.Identity()
            self.downsample = False

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes=10):
        super(ResNet, self).__init__()
        
        self.block = block
        self.channels = [block.expansion * x for x in [64, 128, 256, 512]]

        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layers(block, self.channels[0], self.channels[0], 1, num_layers[0])
        self.layer2 = self._make_layers(block, self.channels[0], self.channels[1], 2, num_layers[1])
        self.layer3 = self._make_layers(block, self.channels[1], self.channels[2], 2, num_layers[2])
        self.layer4 = self._make_layers(block, self.channels[2], self.channels[3], 2, num_layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.channels[-1], num_classes)

    def _make_layers(self, block, in_channel, out_channel, stride, num_layers):
        layers = []
        layers.append(block(in_channel, out_channel, stride=stride))
        for _ in range(1, num_layers):
            layers.append(block(out_channel, out_channel, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
