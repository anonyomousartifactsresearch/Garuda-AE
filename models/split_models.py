import torch
import torch.nn as nn
import torch.nn.functional as F

# We need to import the BasicBlock from your original resnet.py
from .resnet import BasicBlock

# This is the Client-Side Model
class ResNet18_client(nn.Module):
    def __init__(self):
        super(ResNet18_client, self).__init__()
        self.in_planes = 64
        
        # This part is the standard ResNet18 "stem"
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # We only put layer1 on the client
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Client's forward pass
        out = self.relu(self.bn1(self.conv1(x)))
        # Note: Original ResNet puts maxpool here. 
        # For CIFAR10, it's common to skip it, but let's check your resnet.py...
        # Your resnet.py also skips maxpool for CIFAR10.
        # Let's match your original resnet.py:
        
        # --- Matching your models/resnet.py ---
        # out = F.relu(self.bn1(self.conv1(x))) # This is what your file has
        # No maxpool in your original file for CIFAR, so we skip it.
        # ---
        
        # Let's stick to a more standard split that includes maxpool
        # This gives better feature reduction before sending
        out = self.maxpool(out) 
        out = self.layer1(out)
        return out


# This is the Server-Side Model
class ResNet18_server(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_server, self).__init__()
        
        # The 'in_planes' must match the output of the client's layer1
        self.in_planes = 64 * BasicBlock.expansion # which is 64
        
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes) # 512

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Server's forward pass
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out