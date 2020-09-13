import numpy as np
import torch
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import glob
from PIL import Image

from utils import reformat
from transfer import *

"""
Process images in the given folder with a given style
(not camera)
"""

def style_transfer(res, t):
    t1 = time.time()
    res, _ = t.style_net(res)
    t2 = time.time()
    print('{0:.3f} s'.format(t2-t1))
    return res


def get_files_path(path):
    files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]
    for f in files:
        print(f)
    return files


def load_image_as_tensor(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    noise = np.random.randint(-100, 100, img.shape)
    img = img + noise
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.div_(255.)
    return img


if __name__ == '__main__':
    impath = '../imfolder'
    style_path = 'models/state_dict_WAVEWORKING_stylecontent.pth'
    t =  Transfer(10,
                  './video/',
                  './examples/style_img/wave.png',
                  '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-3,
                  1e5, 1e7, 0, 1e-8, gpu=False)

    # loading model
    print('loading state_dict')
    if t.gpu:
        t.style_net.load_state_dict(torch.load(style_path))
    else:
        t.style_net.load_state_dict(torch.load(style_path,  map_location='cpu'))

    # loading IMAGES NAMES
    files = get_files_path(impath)

    # activating gpu mode
    if t.gpu:
        t.style_net = t.style_net.cuda()

    # some params
    width = 640
    height = 360

    for i in range(len(files)):
        print('{}/{}'.format(i, len(files)))

        # get frame
        frame = load_image_as_tensor(files[i], scale=2)
        if t.gpu:
            frame = frame.cuda()

        # get processed image
        output = style_transfer(frame, t)
        output = (1 + output)/2

        # save imgs
        save_image(output, '{}/results/{}_star_{}.jpg'.format(impath, i, 'out'))
        save_image(frame, '{}/results/{}_star_{}.jpg'.format(impath, i, 'in'))
