import numpy as np
import torch
from torchvision import transforms
from configuration import *
import cv2

import io
import zlib
from flask import Flask, request, Response


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

def compress_nparr(nparr):
    """
    https://gist.github.com/andres-fr/f9c0d5993d7e7b36e838744291c26dde
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))
