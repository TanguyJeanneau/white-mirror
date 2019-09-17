import os 
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms

import sys
sys.path.extend(["/usr/local/anaconda3/lib/python3.6/site-packages/",
                 "/home/tanguy/.conda/envs/venv/lib/python3.7/site-packages"])
import cv2


class Dataset(data.Dataset):
    def __init__(self, data_path, img_shape, transform, video_list=None, frame_nb=32):
        self.data_path = data_path
        self.img_shape = img_shape
        self.frame_nb = frame_nb
        self.transform = transform
        if video_list is None:
            self.video_list = os.listdir(data_path)
        else:
            self.video_list = video_list


    def __getitem__(self, i):
        print(self.video_list[i])

        video = cv2.VideoCapture(os.path.join(self.data_path, self.video_list[i]))

        # Chosing a random set of frame_nb consecutive frames
        videolen = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        startframe = np.random.randint(0, max(1, videolen - self.frame_nb))
        endframe = min(videolen, startframe + self.frame_nb)
        # print('len: {}'.format(videolen))
        # print('{} to {}'.format(startframe, endframe))
        # TMP !!
        startframe, endframe = 0, self.frame_nb

        # Skipping frames until startframe
        for i in range(0, startframe): video.read()

        frames = []
        # Processing (endframe - startframe) frames
        for i in range(startframe, endframe):
            ret, frame = video.read()
            if frame is None:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.resize(rgb_frame, self.img_shape)
            
            if self.transform is not None:
                rgb_frame = self.transform(rgb_frame)

            frames.append(rgb_frame)
        video.release()
        return frames


    def __len__(self):
        return len(self.video_list)


def get_loader(batch_size, data_path, img_shape, transform, shuffle=True, video_list=None, frame_nb=32):
    dataset = Dataset(data_path, img_shape, transform, video_list=video_list, frame_nb=frame_nb)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader

if __name__ == '__main__':
    print('start')
    data_path = './video/'
    img_shape=(640, 360)    
    transform = transforms.Compose([transforms.ToTensor()])

    loader = get_loader(1, data_path, img_shape, transform, frame_nb=32)
    for idx, sample in enumerate(loader):
        print('video len: {}'.format(len(sample)))
        print('xxxx')
    print('end')

