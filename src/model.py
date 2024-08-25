import torch.nn as nn
import torch.nn.functional as F

# Residual Block that extacts the features
class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
      super(ResidualBlock, self).__init__()

      conv_block = [
          nn.ReflectionPad2d(1),
          nn.Conv2d(in_channel, in_channel, 3),
          nn.InstanceNorm2d(in_channel),
          nn.ReLU(inplace=True),
          nn.ReflectionPad2d(1),
          nn.Conv2d(in_channel, in_channel, 3),
          nn.InstanceNorm2d(in_channel)
      ]

      self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
      return x + self.conv_block(x)

# Generator of the network
class Generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        net = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # downsampling
        in_channel_ = 64
        out_channel_ = in_channel_ * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channel_, out_channel_, 3, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ]
            in_channel_ = out_channel_
            out_channel_ = in_channel_ * 2

        # residual blocks
        for _ in range(n_residual_blocks):
            net += [ResidualBlock(in_channel_)]

        # upsampling
        out_channel_ = in_channel_ // 2
        for _ in range(2):
            net += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel_, out_channel_, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel_),
                nn.ReLU(inplace=True)
            ]
            in_channel_ = out_channel_
            out_channel_ = in_channel_ // 2

        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channel, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

# Discriminator of the network
class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()

        net = [
            nn.Conv2d(in_channel, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channel = 64
        net += [
            nn.Conv2d(in_channel, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channel = 128
        net += [
            nn.Conv2d(in_channel, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channel = 256
        net += [
            nn.Conv2d(in_channel, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        net += [
            nn.Conv2d(512, 1, 4, padding=1)
        ]

        self.model = nn.Sequential(*net)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
