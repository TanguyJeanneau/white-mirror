import cv2
import numpy as np
import torch
import requests

from transfer import *
from main import style_transfer, reformat

"""
Use laptop camera as continuous image flow,
and process them with pretrained style.
This script uses an API instead of running locally

Need to give a pretrained model (model_path variable)
"""

if __name__ == '__main__':
    post_url = 'http://127.0.0.1:5000/imglive'
    # model_path = 'models/state_dict_STARWORKING_contentandstyle.pth'
    model_path = 'models/state_dict_WAVEWORKING_stylecontent.pth'
    t =  Transfer(10,
                  './video/',
                  './examples/style_img/wave.png',
                  '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-3,
                  1e5, 1e7, 0, 1e-8)
    # loading model
    print('loading state_dict')
    t.style_net.load_state_dict(torch.load(model_path,  map_location='cpu'))
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
    while True:
        # get cam image
        print('frame {}'.format(count))
        check, frame = video.read()

        # HERE, prepare the image and give it to the API.
        # THEN get the processed image and display it

        # get processed image
        # output = torch.from_numpy(np.array([frame]))
        # output = output.type('torch.FloatTensor')
        # output = output/255
        # output = torch.transpose(output, 1,3)
        # output = torch.transpose(output, 3,2)
        # output = style_transfer(output, t)
        # 
        # # formating
        # output = reformat(output)
        h, w, _ = frame.shape
        print(type(frame), frame.shape, frame.min(), frame.max())

        t1 = time.time()
        # h and w are quick fix for undesired image unfolding. To be changed
        req = requests.post(post_url+'?h={}&w={}'.format(h, w), data=frame.tostring())
        t2 = time.time()
        print('REQUETE EFFECTUEE EN {:.2f} s'.format(t2-t1))

        nparr = np.fromstring(req.content, np.uint8)
        # VERY dirty, that's just temporary
        try:
            output = nparr.reshape([h+2, w, 3])
        except ValueError:
            try:
                output = nparr.reshape([h, w, 3])
            except ValueError:
                output = nparr.reshape([h+1, w, 3])
        print(type(output), output.shape, output.min(), output.max())
        
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
    
    
    
