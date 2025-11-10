import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """
    Generates a trigger pattern.
    Output: (batch_size, 3, 32, 32)
    """
    def __init__(self, noise_dim, output_channels=3, img_size=32):
        super(Generator, self).__init__()
        self.init_size = img_size // 4  # Initial size 8x8
        self.l1 = nn.Sequential(nn.Linear(noise_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channels, 3, stride=1, padding=1),
            nn.Tanh(), # Output values between -1 and 1
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        trigger = self.conv_blocks(out)
        return trigger

class Discriminator(nn.Module):
    """
    Classifies "smashed data" (output of client's ResNet layer1).
    Input: (batch_size, 64, 32, 32)
    """
    def __init__(self, input_channels=64, img_size=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 16, bn=False), # 16, 16, 16
            *discriminator_block(16, 32), # 32, 8, 8
            *discriminator_block(32, 64), # 64, 4, 4
            *discriminator_block(64, 128), # 128, 2, 2
        )

        # Classifier
        ds_size = img_size // 2 ** 4 # 32 / 16 = 2
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity