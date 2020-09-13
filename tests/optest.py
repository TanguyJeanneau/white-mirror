import cv2
import numpy as np
# from imageio import imread
from torchvision import transforms

from dataset import get_loader
from utils import reformat


def examine(x, sentence):
    print('***********')
    print(sentence)
    print(x.shape, x.dtype)
    print(x.min(), x.max(), x.mean())


def draw_hsv(flow):
    h, w = flow.shape[:2]
    v, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':

    data_path = '../v3/video/'    
    img_shape = (640, 360)
    videonames = ['output1.mp4', '9_17_s.mp4', '22_26_s.mp4']
    transform = transforms.ToTensor()
    loader = get_loader(1, data_path, img_shape, transform, video_list=videonames, frame_nb=10, shuffle=False)
    
    for idx, frames in enumerate(loader):
        print(idx, len(frames))
        f1, f2 = frames[2], frames[3]
        examine(f1, 'f1 raw')
        f1 = reformat(f1)
        f2 = reformat(f2)
        examine(f1, 'f1 reformated')

        f1 = np.float32(f1)# / 255)
        f2 = np.float32(f2)# / 255)
        examine(f1, 'im1 intermediate')
        examine(f2, 'im2 intermediate')
        im1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
        examine(im1, 'im1 ready')
        examine(im2, 'im2 ready')
        
        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print('OF done!')
        examine(flow, 'flow')

        hsv = draw_hsv(flow)

        examine(hsv, 'hsv final')
        im2w = warp_flow(f1, flow)
        examine(im2w, 'reconstructed image')
        
        cv2.imwrite("{}_flow.jpg".format(idx),hsv)
        cv2.imwrite("{}_im1.jpg".format(idx), f1)
        cv2.imwrite("{}_im2.jpg".format(idx), f2)
        cv2.imwrite("{}_im2w.jpg".format(idx), im2w)
        print('finished')
