import numpy as np
import torch
from torchvision import transforms

sys.path.extend(["/usr/local/anaconda3/lib/python3.6/site-packages/",
                 "/home/tanguy/.conda/envs/venv/lib/python3.7/site-packages"])
import cv2

from utils import reformat
from transfer import *

"""
Process videos in the dataloader with a given style
(not camera)
"""

def style_transfer(res, t):
    t1 = time.time()
    res, _ = t.style_net(res)
    t2 = time.time()
    print('{0:.3f} s'.format(t2-t1))
    return res


if __name__ == '__main__':
    t =  Transfer(10,
                  '../v3/video/',
                  './examples/style_img/wave.png',
                  '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-3,
                  1e5, 1e7, 0, 1e-8, gpu=False)

    # loading model
    print('loading state_dict')
    if t.gpu:
        t.style_net.load_state_dict(torch.load('../state_dict_WAVEWORKING_stylecontent.pth'))
    else:
        t.style_net.load_state_dict(torch.load('../state_dict_WAVEWORKING_stylecontent.pth', map_location='cpu'))
    # t.style_net.load_state_dict(torch.load('model/state_dict_inlearning_styley3.pth'))

    # loading video
    # videonames = ['1_17_s.mp4', '1_3_s.mp4', '3_21_s.mp4', '3_22_s.mp4', '3_23_s.mp4', '10_16_s.mp4', '12_14_s.mp4', '15_25_s.mp4', '25_25_s.mp4', '26_28_s.mp4', 'Neon - 21368.mp4']
    # videonames = ['output1.mp4', '9_17_s.mp4', '22_26_s.mp4']
    videonames = ['9_17_s.mp4']
    # videonames = ['10_21_s.mp4', '28_14_s.mp4', '1_8_s.mp4'] #star
    # videonames = ['Neon - 21368.mp4', '15_25_s.mp4'] #mosaic
    transform = transforms.ToTensor()
    loader = get_loader(1, t.data_path, t.img_shape, transform, video_list=videonames, frame_nb=1440, shuffle=False)
    
    # activating gpu mode
    if t.gpu:
        t.style_net = t.style_net.cuda()

    # some params
    width = 640
    height = 360


    count = 0
    for step, frames in enumerate(loader):
        # open cam
        print('preparing video recording')
        video = cv2.VideoCapture(0)
        video.set(3, width)
        video.set(4, height)
        # prepare recording

        fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
        out = cv2.VideoWriter('output_test{}.mp4'.format(count+8), fourcc, 20.0, (2*width, height))
        print(len(frames))
        for i in range(len(frames)):
            print('{}/{}'.format(i, len(frames)))
            t1 = time.time()

            # get frame
            frame = frames[i]
            if t.gpu:
                frame = frame.cuda()

            # get processed image
            output = style_transfer(frame, t)

            # convert image types
            frame = reformat(frame)
            output = reformat(output)

            # concatenate images
            img = np.concatenate((frame,output),axis=1)

            # save img
            out.write(np.uint8(img))

            t2 = time.time()
            #print('{0:.3f} s'.format(t2-t1))
        
        # cleaning things
        video.release()
        out.release()
        count += 1
