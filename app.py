#!../v2/v2venv/bin/python
from flask import Flask, request, abort, Response, make_response, send_file
import torch
import cv2
import json
import jsonpickle
import io

from transfer import *
from demo_images import load_image_as_tensor, style_transfer
from main import reformat

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
    name_id = 'x'
    if 'img_path' not in request.args:
        return abort(404)
    if 'id' in request.args:
        name_id = request.args['id']
    img_path = request.args['img_path']
    frame = load_image_as_tensor(img_path, scale=2)
    if t.gpu:
        frame = frame.cuda()
    # get processed image
    output = style_transfer(frame, t)
    output = (1 + output)/2
    save_image(output, '../imfolder/results/{}_fromapi_out.jpg'.format(name_id))
    save_image(frame, '../imfolder/results/{}_fromapi_in.jpg'.format(name_id))
    return img_path


@app.route('/imglive', methods=['POST'])
def imglive():
    h = int(request.args['h'])
    w = int(request.args['w'])
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = nparr.reshape([h, w, 3])
    print(type(img), img.shape, img.min(), img.max())

    output = torch.from_numpy(np.array([img]))
    output = output.type('torch.FloatTensor')
    output = output/255
    output = torch.transpose(output, 1,3)
    output = torch.transpose(output, 3,2)
    output = style_transfer(output, t)
    output = reformat(output)
    print(type(output), output.shape, output.min(), output.max())

    response = make_response(output.tostring())
    return response


if __name__ == '__main__':
    app.run(debug=True)
