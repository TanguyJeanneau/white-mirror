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
        t0 = time.time()
        print('frame {}'.format(count))
        # get cam image
        check, frame = video.read()
        h, w, _ = frame.shape

        t1 = time.time()
        print('PRE-REQUEST:  {:.2f}s'.format(t1-t0))
        # h and w are quick fix for undesired image unfolding. To be changed
        req = requests.post(post_url+'?h={}&w={}'.format(h, w), data=frame.tostring())
        t2 = time.time()
        print('REQUEST:      {:.2f} s'.format(t2-t1))

        nparr = np.fromstring(req.content, np.uint8)
        # VERY dirty, that's just temporary
        try:
            output = nparr.reshape([h+2, w, 3])
        except ValueError:
            try:
                output = nparr.reshape([h, w, 3])
            except ValueError:
                output = nparr.reshape([h+1, w, 3])
        
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
        t3 = time.time()
        print('POST-REQUEST: {:.2f}s'.format(t3-t2))
        print('TOTAL FRAME:  {:.2f}s'.format(t3-t0))
        
    # cleaning things
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    
    
