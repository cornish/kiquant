"""
KiNet model architecture for Ki-67 nucleus detection.

Based on: Xing et al. "Pixel-to-pixel Learning with Weak Supervision for
Single-stage Nucleus Recognition in Ki67 Images" IEEE TBME, 2019.

Original code: https://github.com/exhh/KiNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Downsampling block with two convolutions and max pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x  # Return pooled and skip connection


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Ki67Net(nn.Module):
    """
    Ki67Net: U-Net style architecture for Ki-67 nucleus detection and classification.

    Input: RGB image tensor (B, 3, H, W)
    Output: 3-channel voting maps (B, 3, H, W) for:
        - Channel 0: Immunopositive tumor nuclei
        - Channel 1: Immunonegative tumor nuclei
        - Channel 2: Non-tumor nuclei
    """

    def __init__(self, num_cls=3):
        super().__init__()

        # Input transition
        self.input_conv = ConvBNReLU(3, 32)

        # Encoder (downsampling path)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256)
        )

        # Decoder (upsampling path)
        self.up4 = UpBlock(256, 256, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 32)

        # Output transition
        self.output_conv = nn.Conv2d(32, num_cls, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input
        x = self.input_conv(x)

        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # Output
        x = self.output_conv(x)

        return x

    def predict(self, x):
        """Run inference and return numpy array."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output.cpu().numpy()
