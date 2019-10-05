import torch
import copy
import numpy as np
from torch import from_numpy, transpose
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from dataset import get_loader
from utils import reformat
# from transfer import Transfer
from utils import reformat

import sys
sys.path.extend(["/usr/local/anaconda3/lib/python3.6/site-packages/",
                 "/home/tanguy/.conda/envs/venv/lib/python3.7/site-packages"])
import cv2


def examine(x, sentence):
    print('***********')
    print(sentence)
    print(x.shape, x.dtype)
    print(x.min(), x.max(), x.mean())

def warp_flow(img, flow):
    img = np.float32(reformat(img))
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


def array_to_torch(x):
    """ input: np array of shape h, w, 3
        output: pytorch tensor of shape 1, 3, h, w """
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.array([x])
    x = from_numpy(x)
    x = transpose(x, 1,3)
    x = transpose(x, 2,3)
    x = x.div_(255.0)
    return x


def confidence_mask(f1, f2):
    rgb_f, flow_f = opticalflow(f1, f2)
    rgb_b, flow_b = opticalflow(f2, f1)
    f1_w = warp_flow(f1, flow_f)
    f1_w = array_to_torch(f1_w)
    f1_w_w = warp_flow(f1, flow_f + flow_b)
    f1_w_w = array_to_torch(f1_w_w)

    w_w = torch.norm(f1 - f1_w_w, dim=1)**2
    # Parameters to be adjusted
    occlusion_mask = (w_w < 0.01*(torch.norm(f1, dim=1)**2 +
                                  torch.norm(f1_w_w, dim=1)**2))# - 0.005)
    # save_image(occlusion_mask, 'tmp/aocclusion.jpg')
    # save_image(f1_w_w, 'tmp/btest.jpg')
    # save_image(img1, 'tmp/frame1.jpg')
    # save_image(f1_w, 'tmp/frame1_warpedinto2.jpg')
    # save_image(img2, 'tmp/frame2.jpg')
    return occlusion_mask


# input is numpy image array
def opticalflow(img1, img2):
    b, c, h, w = img1.shape
    # examine(img1, 'img1 before reformat')

    img1 = np.float32(reformat(img1))
    img2 = np.float32(reformat(img2))
    # examine(img1, 'img1 after reformat')

    prev = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    nxt = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # examine(prev, 'prev = img1 in grayscale')

    flow = cv2.calcOpticalFlowFarneback(prev, nxt, flow=None,
                                        pyr_scale=0.5, levels=3, # wb 1?
                                        winsize=15, iterations=3,# wb 2?
                                        poly_n=5, poly_sigma=1.2,# wb 1.1?
                                        flags=0)
    examine(flow, 'flow:')

    hsv = np.zeros_like(img1, np.uint8)
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    examine(hsv, 'final hsv:')

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    rgb = np.float32(rgb)
    rgb = array_to_torch(rgb)
    # examine(rgb, 'rgb final')
    return rgb, flow


if __name__ == '__main__':
    
    data_path = '../v3/video/'    
    img_shape = (640, 360)
    # videonames = ['2_26_s.mp4']
    # videonames = ['output1.mp4']
    # videonames = ['Neon - 21368.mp4']
    # videonames = ['output1.mp4', '9_17_s.mp4', '22_26_s.mp4']
    videonames = ['22_26_s.mp4']
    transform = transforms.ToTensor()
    loader = get_loader(1, data_path, img_shape, transform, video_list=videonames, frame_nb=20, shuffle=False)
    t =  Transfer(100,
                  './video/',
                  './examples/style_img/candy.jpg',
                  '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-4,
                  2e-1, 1e0, 0, 0,
                  gpu=True)
    t.style_net.load_state_dict(torch.load('models/state_dict_STARWORKING_contentandstyle.pth', map_location='cpu'))
    
    for idx, frames in enumerate(loader):
        for i in range(5,7):
            # Y'a un truc chelou avec les opticalflows, la deuxieme fois qu'on l'execute f1 deviens blanc ...  a creuser.
            f1, f2 = copy.deepcopy((frames[i-1], frames[i]))

            # Collect optical flow from f1 to f2
            rgb, flow = opticalflow(f1, f2)
            examine(rgb, 'rgb')
            examine(flow, 'flow')
            # Warp f1 to f2
            f1_w = warp_flow(f1, flow)
            f1_w = array_to_torch(f1_w)

            # Compute occlusion mask
            occlusion_mask = confidence_mask(f1, f2)

            # Transfer style to f1, f2, and warp f1 stylized using f1 -> f2 optical flow
            f1_trans = Variable(f1, requires_grad=True)
            f2_trans = Variable(f2, requires_grad=True)
            f1_trans, _ = t.style_net(Variable(f1, requires_grad=True))
            f2_trans, _ = t.style_net(Variable(f2, requires_grad=True))
            f1_trans = 0.5 * (f1_trans + 1)
            f2_trans = 0.5 * (f2_trans + 1)
            f1_trans_w = warp_flow(f1_trans, flow)
            f1_trans_w = array_to_torch(f1_trans_w)

            # Save images for analysis.
            save_image(f1, 'tmp/{}_frame1.jpg'.format(i))
            save_image(f2, 'tmp/{}_frame2.jpg'.format(i))
            save_image(f1_w, 'tmp/{}_frame1warpedinto2.jpg'.format(i))
            save_image(f1_trans, 'tmp/{}_trans_frame1.jpg'.format(i))
            save_image(f2_trans, 'tmp/{}_trans_frame2.jpg'.format(i))
            save_image(f1_trans_w, 'tmp/{}_trans_frame1warpedinto2.jpg'.format(i))
            save_image(rgb, 'tmp/{}_rgb.jpg'.format(i))
            save_image(occlusion_mask, 'tmp/{}__occlusion.jpg'.format(i))
