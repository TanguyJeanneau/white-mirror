import requests
import numpy as np
import time
import cv2
from PIL import Image

from demo_images import get_files_path, load_image_as_tensor
"""
Process images in the given folder with a given style
from API
"""

def truc():
    size=None
    scale=2
    keep_asp=False
    img = Image.open(files[i]).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img)#.transpose(2, 0, 1)
    return img

if __name__ == '__main__':
    impath = '../imfolder/ttt'
    get_url = 'http://127.0.0.1:5000/img?img_path={}&id={}'
    post_url = 'http://127.0.0.1:5000/imglive'
    # loading IMAGES NAMES
    files = get_files_path(impath)

    for i in range(len(files)):
        print('{}/{}'.format(i, len(files)))
        # t1 = time.time()
        # req = requests.get(get_url.format(files[i], i))
        # t2 = time.time()
        # print('{:.2f} s'.format(t2-t1))

        img = truc()
        h, w, _ = img.shape
        print(type(img), img.shape, img.min(), img.max())

        t1 = time.time()
        # h and w are quick fix for undesired image unfolding. To be changed
        req = requests.post(post_url+'?h={}&w={}'.format(h, w), data=img.tostring())
        t2 = time.time()
        print('REQUETE EFFECTUEE EN {:.2f} s'.format(t2-t1))

        nparr = np.fromstring(req.content, np.uint8)
        # VERY dirty, that's just temporary
        try:
            imgout = nparr.reshape([h+2, w, 3])
        except ValueError:
            try:
                imgout = nparr.reshape([h, w, 3])
            except ValueError:
                imgout = nparr.reshape([h+1, w, 3])
        print(type(imgout), imgout.shape, imgout.min(), imgout.max())

        A = Image.fromarray(img)
        A.save("../imfolder/results/{}testinloc.jpg".format(i))
        B = Image.fromarray(imgout)
        B.save("../imfolder/results/{}testoutloc.jpg".format(i))
