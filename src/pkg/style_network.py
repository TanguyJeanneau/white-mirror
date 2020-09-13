import torch
import torchvision 
import torch.nn as nn 
from torch.autograd import Variable
import numpy as np
import sys


def conv_block(name, in_C, out_C, activation='ReLU', kernel_size=3, stride=1, padding=1):
    """
    Convolution block: 
        Convolution layer
        Instance Normalisation
        Activation function (if given)
    """
    block = nn.Sequential()
    block.add_module(name + 'Conv', nn.Conv2d(in_C, out_C, kernel_size, stride, padding))
    block.add_module(name + ' Inst_norm', nn.InstanceNorm2d(out_C))
    if activation == 'ReLU':
        block.add_module(name + ' ' + activation, nn.ReLU(inplace=True))
    elif activation == 'Tanh':
        block.add_module(name + ' ' + activation, nn.Tanh())
    return block


def deconv_block(name, in_C, out_C, activation='ReLU', kernel_size=3, stride=1, padding=1):
    """
    Deconvolution block: 
        Deconvolution layer
        Instance Normalisation
        Activation function (if given)
    """
    block = nn.Sequential()
    block.add_module(name + 'DeConv', nn.ConvTranspose2d(in_C, out_C, kernel_size, stride, padding, output_padding=1))
    block.add_module(name + ' Inst_norm', nn.InstanceNorm2d(out_C))
    if activation == 'ReLU':
        block.add_module(name + ' ' + activation, nn.ReLU(inplace=True))
    elif activation == 'Tanh':
        block.add_module(name + ' ' + activation, nn.Tanh())
    return block


class ResidualBlock(nn.Module):
    """
    Residual block:
        Convolution block
        Convolution block
        Addition
        Activation layer (ReLU)
    """
    def __init__(self, name, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.convblock1 = conv_block(name + '_1', 48, 48)
        self.convblock2 = conv_block(name + '_2', 48, 48, activation='')
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.convblock1(x)
        out = self.convblock2(out)
        out += residual
        out = self.relu(out)
        return out


class ResizeConv(nn.Module):
    """
    Upsample block: 
        Upsample (Bilinear 2x)
        Padding (for size I think)
        Convolution block
            Convolution layer
            Instance Normalisation
            Activation function (ReLU)
    (modification of the deconvolution layer that limit checkerboard patterns) 
    """
    def __init__(self, name, in_C, out_C, kernel_size=3, stride=1, padding=1):
        super(ResizeConv, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv_in = conv_block(name + '_conv', in_C, out_C, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

    def forward(self, x):
        x_in = x
        x_in = nn.functional.interpolate(x_in, mode='bilinear', scale_factor=2)
        out = self.reflection_pad(x_in)
        out = self.conv_in(x_in)
        return out


class StyleNet(nn.Module):
    """
    Style Transfer Network
    3 Convolutions blocks, size/4
    5 Residual blocks
    3 Deconvolution (or Resize-Convolution) blocks, size*4
    """
    def __init__(self):
        super(StyleNet, self).__init__()
        name = "StyleNet"
        deconvolution = False

        self.layer1 = conv_block(name + ' 1', 3, 16, stride=1, padding=1, kernel_size=3)
        self.layer2 = conv_block(name + ' 2', 16, 32, stride=2, padding=1, kernel_size=3)
        self.layer3 = conv_block(name + ' 3', 32, 48, stride=2, padding=1, kernel_size=3)
        self.res1 = ResidualBlock(name + 'ResBlock1', 48, 48)
        self.res2 = ResidualBlock(name + 'ResBlock2', 48, 48)
        self.res3 = ResidualBlock(name + 'ResBlock3', 48, 48)
        self.res4 = ResidualBlock(name + 'ResBlock4', 48, 48)
        self.res5 = ResidualBlock(name + 'ResBlock5', 48, 48)
        if deconvolution:
            self.layer4 = deconv_block(name + ' 4', 48, 32, stride=2, padding=1, kernel_size=3)
            self.layer5 = deconv_block(name + ' 5', 32, 16, stride=2, padding=1, kernel_size=3)
            self.layer6 = conv_block(name + ' 6', 16, 3, stride=1, padding=1, kernel_size=3, activation='Tanh')
        else:
            self.layer4 = ResizeConv(name + ' ResizeConvBlock1', 48, 32)
            self.layer5 = ResizeConv(name + ' ResizeConvBlock2', 32, 16)
            self.layer6 = conv_block(name + ' 6', 16, 3, stride=1, activation='Tanh', padding=1, kernel_size=3)
    
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        res1 = self.res1(out3)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        out4 = self.layer4(res5)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        # # IDK why it was here but unnecessary a priori
        # out6 = 0.5 * (out6 + 1)
        return out6, [out1, out2, out3, res1, res2, res3, res4, res5, out4, out5, out6]


if __name__ == '__main__':
    style_net = StyleNet()
    style_net.load_state_dict(torch.load('outputs/state_dict.pth'))

    rnd = torch.rand(1, 3, 360, 640)
    rnd_output = style_net(rnd)
    torchvision.utils.save_image(rnd, 'outputs/rnd.jpg')
    torchvision.utils.save_image(rnd_output, 'outputs/rndoutput.jpg')
    print(rnd.shape, rnd_output.shape)
    print('ok')
    pass
