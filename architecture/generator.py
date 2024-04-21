import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, base_filters=64):
        super(UNetGenerator, self).__init__()
        self.down1 = EncoderBlock(in_channels, base_filters)
        self.down2 = EncoderBlock(base_filters, base_filters * 2)
        self.down3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.down4 = EncoderBlock(base_filters * 4, base_filters * 8)
        self.bridge = DepthwiseSeparableConv(base_filters * 8, base_filters * 16, kernel_size=3, padding=1)

        self.up4 = DecoderBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up3 = DecoderBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up2 = DecoderBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up1 = DecoderBlock(base_filters * 2 + base_filters, base_filters)

        self.final_conv = nn.Conv2d(base_filters, in_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.down1(x)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        skip4 = self.down4(skip3)
        bridge = self.bridge(skip4)
        up4 = self.up4(bridge, skip4)
        up3 = self.up3(up4, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        return self.final_conv(up1)


# Example usage
generator = UNetGenerator(in_channels=3)
dummy_input = torch.randn(1, 3, 256, 256)
output = generator(dummy_input)
print(output.shape)
