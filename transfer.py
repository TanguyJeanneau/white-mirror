import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import time

from style_network import *
from loss_network import *
from dataset import get_loader
from opticalflow_wip import opticalflow, warp_flow, confidence_mask, array_to_torch


class Transfer:
    def __init__(self, epoch, data_path, style_path, vgg_path, lr, spatial_a, spatial_b, spatial_r, temporal_lambda, gpu=False, img_shape=(640, 360)):
        print('initializing Transfer instance...')
        # General variables
        self.epoch = epoch
        self.data_path = data_path
        self.style_path = style_path
        self.lr = lr
        self.gpu = gpu
        # Model variables
        self.s_a = spatial_a
        self.s_b = spatial_b
        self.s_r = spatial_r 
        self.t_l = temporal_lambda

        # initializing StyleNet...
        self.style_net = StyleNet()
        # initializing LossNet...
        self.loss_net = LossNet(vgg_path)
        self.style_layer = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2']
        # Initializing other losses
        self.content_loss = ContentLoss(self.gpu)
        self.style_loss = StyleLossTest(self.gpu)
        self.temporal_loss = TemporalLoss(self.gpu)
        self.tv_loss = TVLoss(self.gpu)

        # initializing image transformation...
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_shape = img_shape
        print('TRANSFER INIT DONE')

    def load_style(self):
        """ output: style image as torch tensor format
            The image is located in self.style_path
        """
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.open(self.style_path)
        img = img.resize(self.img_shape)
        img = transform(img).float()
        img = Variable(img, requires_grad=True)
        return img

    def train(self):
        """
        THE big badass function. Like, the One.
        """
        print('loading style image')
        style_img = self.load_style()
        style_img = style_img.unsqueeze(0)

        if self.gpu:
            print('activating gpu mode')
            self.style_net = self.style_net.cuda()
            self.loss_net = self.loss_net.cuda()
            style_img = style_img.cuda()
            print('SUPERPOWER LOADED GET READY AF MF')

        adam = optim.Adam(self.style_net.parameters(), lr=self.lr)
        
        print('loading data')
        loader = get_loader(1, self.data_path, self.img_shape, self.transform, frame_nb=16+1)

        # Save losses under .csv file
        with open('loss.csv', 'w') as f:
            f.write('content,style,spatial,temporal,total\n')

        # Idk why it's here but I don't wanna mess it up
        time.sleep(1)
        # This can be put out of the loop cause wont change through time
        s = self.loss_net(style_img, self.style_layer)
        print("Let's go")
        for count in range(self.epoch):
            t1 = time.time()
            vidnb = 0
            for step, frames in enumerate(loader):
                vidnb += 1
                lf = len(frames)
                # print(len(frames))
                # print(frames[0].shape)
                for i in range(1, lf):
                    # THIS. IS. NE-CE-FU-CKING-SSA-RY
                    adam.zero_grad()

                    # load frames
                    x_t = Variable(frames[i], requires_grad=True)
                    x_t1 = Variable(frames[i-1], requires_grad=True)
                    # print('1')
                    if self.gpu:
                        x_t = x_t.cuda()
                        x_t1 = x_t1.cuda()

                    # compute outputs
                    h_xt, allimgs = self.style_net(x_t)
                    h_xt1, _ = self.style_net(x_t1)
                    # print('2')

                    # calculate loss network outputs
                    s_xt = self.loss_net(x_t, self.style_layer)
                    s_xt1 = self.loss_net(x_t1, self.style_layer)
                    s_hxt = self.loss_net(h_xt, self.style_layer)
                    s_hxt1 = self.loss_net(h_xt1, self.style_layer)
                    # print('3')

                    # calculate content loss
                    # why s_xt[3] => content features from conv4_2 layer
                    content_t = self.content_loss(s_xt[3], s_hxt[3])
                    content_t1 = self.content_loss(s_xt1[3], s_hxt1[3])
                    content_loss = content_t + content_t1
                    # print('4')

                    # calculate style loss
                    style_t = 0
                    style_t1 = 0
                    for k in range(len(self.style_layer)):
                        # Homemade: give more importance to the deepest layers
                        # In the paper it's always 1
                        coef = 1 + 0.33 * k
                        style_t += coef * self.style_loss(s[k], s_hxt[k])
                        style_t1 += coef * self.style_loss(s[k], s_hxt1[k])
                    style_loss = style_t + style_t1
                    # print('5')

                    # # calculate total varation loss
                    # # to FIX AND OPTIMIZE
                    # # https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
                    # # So far I haven't included it here,
                    # # not really needed but in the paper
                    # tv_loss = self.tv_loss(x_t)

                    # calculate optical flow
                    # TO FIX
                    rgb, flow = opticalflow(h_xt, h_xt1)
                    h_xt_w = warp_flow(h_xt, flow)
                    h_xt_w = array_to_torch(h_xt_w)
                    h_xt_w = h_xt_w.to(device=0)
                    occlusion_mask = confidence_mask(h_xt, h_xt1)

                    if self.gpu:
                       occlusion_mask = occlusion_mask.cuda()
                    # print('6')

                    # calculate temporal loss
                    temporal_loss = self.temporal_loss(h_xt1, h_xt_w, occlusion_mask)

                    # putting it all together
                    spatial_loss =  self.s_b * style_loss + self.s_a * content_loss #+ self.s_r * tv_loss 
                    Loss = spatial_loss + self.t_l * temporal_loss 
                    # print('7')

                    # Optimization
                    Loss.backward(retain_graph=True)
                    adam.step()
                    # print('8')

                    # printing stuff
                    print('{}/{} frame'.format(i, lf))
                    test = 0
                    for name, param in self.style_net.named_parameters():
                        if param.requires_grad:
                           if 'weight' in name:
                               ci, co, k1, k2 = param.data.shape
                               # print('{0:.3e} {1}'.format(torch.std(param.data)/(ci*co*k1*k2), name))
                           test += torch.abs(param.data.sum()).item()
                    print('sum of all weights: {0:.3e}'.format(test)) # Had some issues with that
                    print('content:  {0:.3e} ({1:.2f}%)'.format(content_loss.item(), 100*self.s_a*content_loss/Loss))
                    print('style:    {0:.3e} ({1:.2f}%)'.format(style_loss.item(), 100*self.s_b*style_loss/Loss))
                    print('temporal: {0:.3e} ({1:>.2f}%)'.format(temporal_loss.item(), 100*self.t_l*temporal_loss/Loss))
                    # print('tv:       {0:.3e} ({1:.2f}%)'.format(tv_loss.item(), 100*self.s_r*tv_loss/Loss))
                    print('total:    {0:.3e}'.format(Loss.item()))

                # # print the memory and cache use, useful in case of OOM error
                # print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_cached(0), torch.cuda.max_memory_cached(0)))
                # print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_allocated(0), torch.cuda.max_memory_allocated(0)))
                print('XXXXXXXXXXXXXXXXXXXXXXXXX')
                # Save some layers outputs for debugging purpose
                for j in range(len(allimgs)):
                    layer = allimgs[j]
                    save_image(layer[:,0,:,:], 'outputs/test3_{}.jpg'.format(j))

                # # Saving loss as .csv file
                # newline='{},{},{},{},{}\n'.format(content_loss,
                #                                      style_loss,
                #                                    #  tv_loss,
                #                                      spatial_loss,
                #                                      temporal_loss, Loss)
                # with open('loss.csv', 'a') as f:
                #     f.write(newline)

                # # print gradients, for debugging purpose
                # for p in self.style_net.parameters():
                #     ttt = torch.abs(p.grad).mean()
                #     print('==gradient:{0:.3e}'.format(ttt))

                # Save current model state and output
                save_image(torch.stack([x_t[0], h_xt[0]]), 'outputs/test3.jpg')
                torch.save(self.style_net.state_dict(), 'outputs/state_dict.pth')
                print('XXX img & model saved XXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXX')

            # Save model and layes output at the end of each epoch
            for j in range(len(allimgs)):
                layer = allimgs[j]
                save_image(layer[:,0,:,:], 'outputs/layer_{}_epoch_{}.jpg'.format(j, count))
            save_image(torch.stack([x_t[0], h_xt[0]]), 'outputs/layer_epoch_{}.jpg'.format(count))
            torch.save(self.style_net.state_dict(), 'outputs/state_dict_end_of_epoch_{}.pth'.format(count))

            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XX    END OF EPOCH {}'.format(count))
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            t2 = time.time()
            print('XX  TOTAL DURATION {}s'.format(t2-t1))
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')


if __name__ == '__main__':
    # torch.cuda.device(0)

    print('loading')
    # Init transfer class
    t =  Transfer(100,
                  '/root/white-mirror/video/',
                  './examples/style_img/wave.png',
                  '/root/.torch/models/vgg19-dcbb9e9d.pth',
                  1e-4,
                  5e-1, 1e0, 0, 1e5,
                  gpu=True)  # Here to switch CPU/GPU

    # Load pretrained style
    # print('loading pretrained style...')
    # t.style_net.load_state_dict(torch.load('model/state_dict_STARWORKING_contentandstyle.pth'))

    # some infos
    device_nb = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_nb)
    if t.gpu:
        print('GPU processing ENABLED on {}'.format(device_name))
    else:
        print('only processing on CPU...')

    # # Print current GPU memory use, in case of OOM
    # print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_cached(0), torch.cuda.max_memory_cached(0)))
    # print('{0:.3e} / {1:.3e}'.format(torch.cuda.memory_allocated(0), torch.cuda.max_memory_allocated(0)))

    # Let's go !
    print('go')
    t.train()
