import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import vgg
from utils import print_memory_usage, reformat
from transfer import *


def vgg19(vgg_path, pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print('xxxxCreating VGG19 model...')
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg.VGG(vgg.make_layers(vgg.cfg['E']), **kwargs)
    if pretrained:
        state_dict = torch.load(vgg_path)
        state_dict = {k:v for k, v in state_dict.items() if 'class' not in k}
        model.load_state_dict(state_dict)
    return model


class LossNet(nn.Module):
    def __init__(self, vgg_path):
        super(LossNet, self).__init__()
        print('xxInitializing LossNet...')
        tempvgg19 = vgg19(vgg_path)
        # tempvgg19 = vgg.vgg19_bn(pretrained=True)
        model_list = list(tempvgg19.features.children())
        self.conv1_1 = model_list[0]
        self.conv1_2 = model_list[2]
        self.conv2_1 = model_list[5]
        self.conv2_2 = model_list[7]
        self.conv3_1 = model_list[10]
        self.conv3_2 = model_list[12]
        # yeah ok right this isn't the cleanest
        self.conv3_3 = model_list[14]
        self.conv3_4 = model_list[16]
        self.conv4_1 = model_list[19]
        self.conv4_2 = model_list[21]
        self.conv4_3 = model_list[23]
        self.conv4_4 = model_list[25]
        self.conv5_1 = model_list[28]
        self.conv5_2 = model_list[30]
        self.conv5_3 = model_list[32]
        self.conv5_4 = model_list[34]
        print('xxLOSSNET INIT DONE')

    def normalize_for_vgg(self, x, gpu = True):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if gpu:
            mean = mean.cuda()
            std = std.cuda()
        return (x - mean) / std

    def forward(self, x, out_key, gpu = True):
        x = self.normalize_for_vgg(x, gpu)
        out = {}
        out['conv1_1'] = F.relu(self.conv1_1(x))
        out['conv1_2'] = F.relu(self.conv1_2(out['conv1_1']))
        out['pool1']  = F.max_pool2d(out['conv1_2'], kernel_size=2, stride=2)

        out['conv2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['conv2_2'] = F.relu(self.conv2_2(out['conv2_1']))
        out['pool2']  = F.max_pool2d(out['conv2_2'], kernel_size=2, stride=2)
        
        out['conv3_1'] = F.relu(self.conv3_1(out['pool2']))
        # Could've done it better but you know time is money
        out['conv3_2'] = F.relu(self.conv3_2(out['conv3_1']))
        out['conv3_3'] = F.relu(self.conv3_3(out['conv3_2']))
        out['conv3_4'] = F.relu(self.conv3_4(out['conv3_3']))
        out['pool3']  = F.max_pool2d(out['conv3_4'], kernel_size=2, stride=2)

        out['conv4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['conv4_2'] = F.relu(self.conv4_2(out['conv4_1']))
        out['conv4_3'] = F.relu(self.conv4_3(out['conv4_2']))
        # And I ain't have no money
        out['conv4_4'] = F.relu(self.conv4_4(out['conv4_3']))
        out['pool4']  = F.max_pool2d(out['conv4_4'], kernel_size=2, stride=2)

        out['conv5_1'] = F.relu(self.conv5_1(out['pool4']))
        out['conv5_2'] = F.relu(self.conv5_2(out['conv5_1']))
        out['conv5_3'] = F.relu(self.conv5_3(out['conv5_2']))
        # F
        out['conv5_4'] = F.relu(self.conv5_4(out['conv5_3']))
        out['pool5']   = F.max_pool2d(out['conv5_4'], kernel_size=2, stride=2)

        return [out[key] for key in out_key]


class ContentLoss(nn.Module):
    def __init__(self, gpu):
        super(ContentLoss, self).__init__()

    def forward(self, x, target):
        b, c, h, w = x.shape
        return (1 /(c * h * w)) * torch.pow((x - target), 2).sum()

class StyleLoss(nn.Module):
    def __init__(self, gpu):
        super(StyleLoss, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, target):
        # channel = x.shape[3]  # 1 isn't it ??
        channel = x.shape[1]
        return ( 1 / channel ** 2) * self.loss(GramMatrix()(x), GramMatrix()(target))

class StyleLossTest(nn.Module):
    def __init__(self, gpu):
        super(StyleLossTest, self).__init__()

    def forward(self, x, target):
        channel = x.shape[1]
        frobsqd = torch.pow(torch.abs(GramMatrix()(x) - GramMatrix()(target)), 2).sum()
        frob = (1 / channel ** 2) * frobsqd
        return frob

class TemporalLoss2(nn.Module):
    """
    x: frame t 
    x1: frame t-1
    """
    def __init__(self, gpu):
        super(TemporalLoss2, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, x1):
        x = x.view(1, -1)
        x1 = x1.view(1, -1)
        
        D = x1.shape[1]
        diffs = torch.pow((x - x1), 2)
        return (1 / D) * diffs.sum()#, f_x1

class TemporalLoss(nn.Module):
    """
    TO BE FIXED, like this isnt working at all
    x: frame t 
    f_x1: optical flow(frame t-1)
    cm: confidence mask of optical flow 
    """
    def __init__(self, gpu):
        super(TemporalLoss, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, f_x1, cm):
        x = x.view(1, -1)
        f_x1 = f_x1.view(1, -1)
        cm2 = torch.stack([cm, cm, cm])
        cm2 = cm2.view(1, -1)
        # cm = cm.view(-1)
        
        D = f_x1.shape[1]
        diffs = torch.pow((x - f_x1), 2)
        return (1 / D) * (cm2 * diffs).sum()#, f_x1


class TVLoss(nn.Module):
    def __init__(self, gpu):
        super(TVLoss,self).__init__()

    def forward(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        # h_tv = torch.pow((x[:,:,1:,1:]-x[:,:,:-1,1:]),2).sum(1)
        # w_tv = torch.pow((x[:,:,1:,1:]-x[:,:,1:,:-1]),2).sum(1) 
        # tot_tv = 2* torch.pow((h_tv/count_h + w_tv/count_w), 1/2).sum()
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:-1]),2).sum() 
        tot_tv = 2* (h_tv/count_h + w_tv/count_w)
        return tot_tv

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

                    
class GramMatrix(nn.Module):
    """
    Gram Matrix, you know. It's in the name.
    """
    def forward(self, x):
        b, c, h, w = x.shape
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return 1/ (h * w) * G

if __name__ == '__main__':
    lossnet = LossNet(VGG_PATH)
