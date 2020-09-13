import cv2
import numpy as np
import torch

import time
from transfer import *
from configuration import *

"""
Use laptop camera as continuous image flow,
and process them with pretrained style.

Need a pretrained VGG model for feature extraction (model_path variable)
"""

def reformat(img):
    """ input: torch tensor, shape (b, c, h, w)
        output: the first of the batch under np image format, shape (h, w, c)
        So far, this function has only been used for batch size =1
    """
    b, c, h, w = img.shape
    # So far b=1
    img = img[0]
    # Switch dimensions
    img = torch.transpose(img, 0,2)
    img = torch.transpose(img, 0,1)
    img = img.cpu().detach().numpy()
    # Image format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Scale
    img = np.uint8(((img + 1) / 2 * 255))
    return img


def style_transfer(img, t):
    """ input: img: torch tensor for image batch. So far, b=1 
               t: Transfer object
        output: style transfer of img
    """
    t1 = time.time()
    res, _ = t.style_net(img)
    t2 = time.time()
    print('{0:.3f} s'.format(t2-t1))
    return res

if __name__ == '__main__':
    model_path = 'models/state_dict_STARWORKING_contentandstyle.pth'
    t =  Transfer(10,
                  VIDEO_PATH,
                  './examples/style_img/wave.png',
                  VGG_PATH,
                  1e-3,
                  1e5, 1e7, 0, 1e-8, gpu=GPU)
    # loading model
    print('loading state_dict')
    t.style_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    # some params
    width = 640
    height = 360
    # open cam
    print('reparing video recording')
    video = cv2.VideoCapture(0)
    video.set(3, width)
    video.set(4, height)
    # prepare recording
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
    count = 0
    nb_iter = 0
    avg = 0
    while True:
        # get cam image
        print('frame {}'.format(count))
        check, frame = video.read()

        # get processed image
        output = torch.from_numpy(np.array([frame]))
        output = output.type('torch.FloatTensor')
        output = output/255
        output = torch.transpose(output, 1,3)
        output = torch.transpose(output, 3,2)
        ti = time.time()
        output = style_transfer(output, t)
        d= time.time() -ti
        avg = (avg* nb_iter + d) / (nb_iter +1)
        nb_iter +=1
        print(avg, d)
        
        # formating
        output = reformat(output)

        # concatenate images
        img = np.concatenate((frame,output),axis=1)

        # display images
        cv2.imshow("test", img)

        # save img
        out.write(img)

        # controling the exit condition
        key = cv2.waitKey(1)
        if key == ord('q') :
            break
        count += 1
        
    # cleaning things
    video.release()
    out.release()
    cv2.destroyAllWindows()

