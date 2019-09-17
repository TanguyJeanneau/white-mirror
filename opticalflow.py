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

# input is numpy image array
def opticalflow1(img1, img2):
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
    # examine(flow, 'flow:')

    hsv = np.zeros_like(img1, np.uint8)
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # examine(hsv, 'final hsv:')

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    rgb = np.float32(rgb)
    # examine(rgb, 'rgb:')

    f_img1 = warp_flow(img1, flow)
    # examine(f_img1, 'img1 warped to img2 with flow')

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # examine(gray, 'gray:')

    gm = np.where(gray > 10, np.ones_like(gray), np.zeros_like(gray))
    gm = from_numpy(gm)
    
    f_img1 = array_to_torch(f_img1)
    rgb = array_to_torch(rgb)
    # examine(rgb, 'rgb final')
    return rgb, gm, f_img1


if __name__ == '__main__':
    
    data_path = '../v3/video/'    
    img_shape = (640, 360)
    videonames = ['2_26_s.mp4']
    # videonames = ['output1.mp4', '9_17_s.mp4', '22_26_s.mp4']
    transform = transforms.ToTensor()
    loader = get_loader(1, data_path, img_shape, transform, video_list=videonames, frame_nb=10, shuffle=False)
    
    for idx, frames in enumerate(loader):
        # Y'a un truc chelou avec les opticalflows, ce la deuieme fois qu'on l'execute f1 deviens blanc ...  a creuser.
        f1, f2 = frames[2], frames[3]
        # examine(f1, 'f1 raw')

        rgb, gm, f_img1 = opticalflow1(f1, f2)
        save_image(f1, '{}_opflow1_frame1.jpg'.format(idx))
        save_image(f2, '{}_opflow1_frame2.jpg'.format(idx))
        save_image(rgb, '{}_opflow1_rgb.jpg'.format(idx))
        save_image(gm, '{}_opflow1_gm.jpg'.format(idx))
        save_image(f_img1, '{}_opflow1_frame2warp.jpg'.format(idx))
