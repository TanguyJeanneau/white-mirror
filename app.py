#!../v2/v2venv/bin/python
from flask import Flask, request, abort
import torch

from transfer import *
from demo_images import load_image_as_tensor, style_transfer

app = Flask(__name__)

# model_path = 'models/state_dict_WAVEWORKING_stylecontent.pth'
model_path = 'models/state_dict_STARWORKING_contentandstyle.pth'
t =  Transfer(10,
              './video/',
              './examples/style_img/wave.png',
              '/home/tfm/.torch/models/vgg19-dcbb9e9d.pth',
              1e-3,
              1e5, 1e7, 0, 1e-8)
# load style
t.style_net.load_state_dict(torch.load(model_path,  map_location='cpu'))
# some params
width = 640
height = 360


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/test', methods=['GET'])
def test():
    print(request.args)
    if 'type' in request.args:
        typ = request.args['type']
        if typ == 'data_path':
            return t.data_path
        if typ == 'style_net':
            return str(t.style_net)
    return 'nope'


@app.route('/img')
def img():
    if 'img_path' not in request.args:
        return abort(404)
    img_path = request.args['img_path']
    frame = load_image_as_tensor(img_path, scale=2)
    if t.gpu:
        frame = frame.cuda()
    # get processed image
    output = style_transfer(frame, t)
    output = (1 + output)/2
    save_image(output, '../imfolder/results/fromapi_out.jpg')
    save_image(frame, '../imfolder/results/fromapi_in.jpg')
    return img_path


@app.route('/imglive')
def imglive():
    # WIP
    return False


if __name__ == '__main__':
    app.run(debug=True)
