#!../venv/bin/python
from flask import Flask, request, abort, Response, make_response, send_file
import torch
import cv2
import json
import io

from transfer import *
from demo_images import load_image_as_tensor
from main import style_transfer, reformat

app = Flask(__name__)

# model_path = 'models/state_dict_WAVEWORKING_stylecontent.pth'
model_path = 'models/state_dict_STARWORKING_contentandstyle.pth'
t =  Transfer(10,
              './video/',
              './examples/style_img/wave.png',
              '/root/.torch/models/vgg19-dcbb9e9d.pth',
              1e-3,
              1e5, 1e7, 0, 1e-8,
              gpu=True)
# load style
t.style_net.load_state_dict(torch.load(model_path))
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
    # output = (1 + output)/2
    # save_image(output, '../imfolder/results/{}_fromapi_out.jpg'.format(name_id))
    # save_image(frame, '../imfolder/results/{}_fromapi_in.jpg'.format(name_id))
    return img_path


@app.route('/imglive', methods=['POST'])
def imglive():
    t1_1 = time.time()
    h = int(request.args['h'])
    w = int(request.args['w'])
    t1_2 = time.time()
    print(t1_2-t1_1)
    r = request
    t1_3 = time.time()
    print(t1_3-t1_2)
    nparr = np.frombuffer(r.data, dtype=np.uint8)
    t1_4 = time.time()
    print(t1_4-t1_3)
    img = nparr.reshape([h, w, 3])
    t2 = time.time()
    print(t2-t1_4)
    print('request unpacking:  {:.2f}s'.format(t2-t1_1))

    output = torch.from_numpy(np.array([img]))
    output = output.type('torch.FloatTensor')
    output = output/255
    output = torch.transpose(output, 1,3)
    output = torch.transpose(output, 3,2)
    t3 = time.time()
    print('pre-processing img: {:.2f}s'.format(t3-t2))

    output, _ = t.style_net(output)
    output = reformat(output)
    t4 = time.time()
    print('image processing:   {:.2f}s'.format(t4-t3))
    print('total request:      {:.2f}s'.format(t4-t1_1))

    response = make_response(output.tobytes())
    return response


if __name__ == '__main__':
    app.run(debug=True, port='8080')
