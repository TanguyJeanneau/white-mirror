import copy
import numpy as np
from torch import from_numpy, transpose
from torchvision import transforms
from torchvision.utils import save_image

from dataset import get_loader
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


def confidence_mask(img1, img2):
    opflow_fwd = opticalflow(img1, img2)
    opflow_bwd = opticalflow(img2, img1)

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

    f_img1 = warp_flow(img1, flow)
    f_img1 = array_to_torch(f_img1)
    return rgb, flow, f_img1


if __name__ == '__main__':
    
    data_path = '../v3/video/'    
    img_shape = (640, 360)
    # videonames = ['2_26_s.mp4']
    videonames = ['output1.mp4']
    # videonames = ['Neon - 21368.mp4']
    # videonames = ['output1.mp4', '9_17_s.mp4', '22_26_s.mp4']
    transform = transforms.ToTensor()
    loader = get_loader(1, data_path, img_shape, transform, video_list=videonames, frame_nb=20, shuffle=False)
    
    for idx, frames in enumerate(loader):
        for i in range(5,7):
            # Y'a un truc chelou avec les opticalflows, ce la deuieme fois qu'on l'execute f1 deviens blanc ...  a creuser.
            f1, f2 = copy.deepcopy((frames[i-1], frames[i]))
            # examine(f1, 'f1 raw')

            flow, raw, f_img1 = opticalflow(f1, f2)
            raw = np.array([raw])
            raw = from_numpy(raw)
            examine(raw, 'refined raw flow:')
            # f1 = np.float32(reformat(f1))
            # f_img1 = warp_flow(f1, flow)
            # f_img1 = array_to_torch(f_img1)

            save_image(f1, 'tmp/{}_opflow1_frame1.jpg'.format(i))
            save_image(f2, 'tmp/{}_opflow1_frame2.jpg'.format(i))
            save_image(f_img1, 'tmp/{}_opflow1_frame2warp.jpg'.format(i))
            save_image(flow, 'tmp/{}_opflow1_flow.jpg'.format(i))
            save_image(raw[..., 0], 'tmp/{}_opflow1_flowraw0.jpg'.format(i))
            save_image(raw[..., 1], 'tmp/{}_opflow1_flowraw1.jpg'.format(i))
