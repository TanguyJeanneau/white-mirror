"""
OBSOLETE
File I used for my Master Thesis Defence
Do NOT delete please
"""

import cv2
import numpy as np
import torch

from transfer import *

def rotate(img, size=None):
    """
    input: numpy image of shape h, w, c
    output: same image rotated of 180 degrees"""
    h, w, c = img.shape
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, 180, 1)
    rotatedimg = cv2.warpAffine(img, M, (w, h))
    if size is not None:
        rotatedimg = cv2.resize(rotatedimg, size)
    return rotatedimg


def reformat(img):
    b, c, h, w = img.shape
    img = img[0]
    img = torch.transpose(img, 0,2)
    img = torch.transpose(img, 0,1)
    img = img.cpu().detach().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.uint8(img * 255)
    return img


def style_transfer(img, t):
    t1 = time.time()
    res, _ = t.style_net(img)
    t2 = time.time()
    print('{0:.3f} s'.format(t2-t1))
    return res


def load_t_with_style(style_path):
    t =  Transfer(10,
                  './video/',
                  './examples/style_img/wave.png',
                  '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-3,
                  1e5, 1e7, 0, 1e-8)
    # loading model
    print('loading state_dict')
    t.style_net.load_state_dict(torch.load(style_path,  map_location='cpu'))
    return t


def load_image(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    # some params
    cam_width = 640
    cam_height = 360
    width = 960
    height = 540

    # load objects Transfer and associated style
    style_list = ['../state_dict_WAVEWORKING_stylecontent.pth',
                 '../state_dict_STARWORKING_contentandstyle.pth',
                 '../state_dict_UDNIEWORKING_c4e-1s1_E19.pth']
    style_img_list = ['../v3/examples/style_img/wave.png',
                      '../v3/examples/style_img/star.png',
                      '../v3/examples/style_img/udnie.jpg']
    t_list = []
    img_list = []
    for i in range(len(style_list)):
        t_list.append(load_t_with_style(style_list[i])) 
        img_list.append(load_image(style_img_list[i], size=int(width/4), keep_asp=True))
        print(img_list[i].shape)

    # open cam
    print('reparing video recording')
    video = cv2.VideoCapture(0)
    video.set(3, cam_width)
    video.set(4, cam_height)

    # some variable initialization
    count = 0
    whichstyle = 0
    smallheight, smallwidth, _ = img_list[whichstyle].shape
    while True:
        # get cam image
        print('frame {}'.format(count))
        check, frame = video.read()

        # pre-process image
        output = torch.from_numpy(np.array([frame]))
        output = output.type('torch.FloatTensor')
        output = output.div_(255.)
        output = torch.transpose(output, 1,3)
        output = torch.transpose(output, 3,2)

        # apply style transfer
        output = style_transfer(output, t_list[whichstyle])
        
        # format output
        output = reformat(output)
        output = rotate(output, size=(width, height))

        # adding style image in the top left corner
        output[:smallheight, :smallwidth, :] = img_list[whichstyle]

        # # concatenate images
        # img = np.concatenate((frame,output),axis=1)
        # frame = rotate(frame, size=(width, height))
        img = output

        # display image(s)
        cv2.imshow("TFM", img)

        # exit condition and style swap
        key = cv2.waitKey(1)
        if key == ord('q') :
            break
        if key == ord('n'):
            whichstyle = (whichstyle + 1) % len(t_list)
            smallheight, smallwidth, _ = img_list[whichstyle].shape
            print('new shape: {} / {}'.format(smallheight, smallwidth))

        count += 1
        
    # cleaning things
    video.release()
    cv2.destroyAllWindows()
