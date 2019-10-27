import numpy as np
import torch
from torchvision import transforms
import sys

sys.path.extend(["/usr/local/anaconda3/lib/python3.6/site-packages/",
                 "/home/arthur/.conda/envs/venv/lib/python3.7/site-packages"])
import cv2


def reformat(img, float_to_int=True):
    """ from a pytorch tensor of shape b, c, h, w
    to an identical numpy array of shape  h, w, c"""
    b, c, h, w = img.shape
    img = img[0]
    img = torch.transpose(img, 0,2)
    img = torch.transpose(img, 0,1)
    img = img.cpu().detach().numpy()
    if float_to_int:
        img = np.uint8(255 * img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def print_memory_usage():
    print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_cached(0),
                                     torch.cuda.max_memory_cached(0)))
    print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_allocated(0),
                                     torch.cuda.max_memory_allocated(0)))
