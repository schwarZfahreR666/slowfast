import os
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import params
import math
import random
from torch.utils.data import DataLoader

class Image_Dataset(Dataset):
    def __init__(self, train_path , clip_len=16,action='train',frame_sample_rate=1):
        
        root_dir = 'video'
        self.clip_len=clip_len
        file_s=[]
        label_s=[]
        self.file_list=[]
        self.label_list=[]
        self.action = action
        self.clip_len = clip_len

        self.short_side = [128, 160]
        self.crop_size = 99
        self.frame_sample_rate = frame_sample_rate
        with open(train_path,'r') as f:
            datas = f.readlines()
            for data in datas:
                num = data.split(':')[0]
                label = data.split(':')[1].replace('\n','')
                label = label.replace(' ','')
                path = root_dir + '/' + num
                file_s.append(path)
                label_s.append(label)
            
        label_s = np.array(label_s).astype(np.int64)
        
        self.file_list=file_s
        self.label_list=label_s
    def __getitem__(self, index):
        imgs=sorted(os.listdir(self.file_list[index]))
        buffer = []
        # self.frame_sample_rate = math.ceil(len(imgs)/params['sample_len'])

        for img_index in range(0, len(imgs)):
            if img_index % self.frame_sample_rate != 0:
                continue
            img_path = os.path.join(self.file_list[index],imgs[img_index])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            buffer.append(img_rgb)
        buffer = np.array(buffer, np.dtype('float32'))

        buffer = self.random_erasing(buffer)
        
        # if self.action == 'train':
        #     buffer = self.randomflip(buffer) # 训练时随机翻转
        buffer = self.crop(buffer, self.clip_len, self.crop_size) # 随机选择开始位置和图像中位置
        buffer = self.normalize(buffer) # 归一化
        buffer = self.to_tensor(buffer) # [D,H,W,C] -> [C,D,H,W]符合 Pytorch格式

        label=self.label_list[index]
        return buffer,label
    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size): # 随机选择开始位置和图像中的框
        # randomly select time index for temporal jittering
        time_dif = buffer.shape[0] - clip_len

        if time_dif > 0:
            time_index = np.random.randint(time_dif)
        else:
            time_index = 0
            pading = np.zeros(((-time_dif)+1, buffer.shape[1], buffer.shape[2], buffer.shape[3]))
            pading = pading.astype(np.dtype('float32'))
            buffer = np.append(buffer, pading, axis=0)

        # Randomly select start indices in order to crop the video
        height_dif = buffer.shape[1] - crop_size
        height_index = np.random.randint(height_dif) if (height_dif > 0) else 0
        width_dif = buffer.shape[2] - crop_size
        width_index = np.random.randint(width_dif) if (width_dif > 0) else 0

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        # buffer = buffer[time_index:time_index + clip_len, :, :, :]

        return buffer  
    
    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer
    
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def random_erasing(self, buffer):
        for i, frame in enumerate(buffer):
            buffer[i] = self.eraser(frame)
        return buffer

    def eraser(self, input_img):
        p = 0.5
        s_l = 0.02
        s_h = 0.2
        r_1 = 0.3
        r_2 = 1 / 0.3
        v_l = 0
        v_h = 255
        pixel_level = False
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img


            
    def __len__(self):

        return len(self.file_list)
    def deal_with():
        pass


class Test_Dataset(Dataset):
    def __init__(self, train_path, clip_len=16, action='train', frame_sample_rate=1):

        root_dir = 'video'
        self.clip_len = clip_len
        file_s = []
        label_s = []
        self.file_list = []
        self.label_list = []
        self.action = action
        self.clip_len = clip_len

        self.short_side = [128, 160]
        self.crop_size = 99
        self.frame_sample_rate = frame_sample_rate
        with open(train_path, 'r') as f:
            datas = f.readlines()
            for data in datas:
                num = data.replace('\n', '')
                # num = data.split(':')[0]
                # label = data.split(':')[1].replace('\n', '')
                # label = label.replace(' ', '')
                path = root_dir + '/' + num
                file_s.append(path)
                # label_s.append(label)

        # label_s = np.array(label_s).astype(np.int64)

        self.file_list = file_s
        # self.label_list = label_s

    def __getitem__(self, index):
        imgs = sorted(os.listdir(self.file_list[index]))
        buffer = []
        for img_index in range(0, len(imgs)):
            img_path = os.path.join(self.file_list[index], imgs[img_index])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            buffer.append(img_rgb)
        buffer = np.array(buffer, np.dtype('float32'))


        # if self.action == 'train':
        #     buffer = self.randomflip(buffer) # 训练时随机翻转
        #buffer = self.crop(buffer, self.clip_len, self.crop_size)  # 随机选择开始位置和图像中位置
        buffer = self.normalize(buffer) # 归一化
        buffer = self.to_tensor(buffer)  # [D,H,W,C] -> [C,D,H,W]符合 Pytorch格式

        return buffer, self.file_list[index]

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def crop(self, buffer, clip_len, crop_size):  # 随机选择开始位置和图像中的框
        # randomly select time index for temporal jittering
        time_dif = buffer.shape[0] - clip_len
        if time_dif > 0:
            time_index = np.random.randint(time_dif)
        else:
            time_index = 0
            pading = np.zeros(((-time_dif) + 1, buffer.shape[1], buffer.shape[2], buffer.shape[3]))
            pading = pading.astype(np.dtype('float32'))
            buffer = np.append(buffer, pading, axis=0)
        # Randomly select start indices in order to crop the video
        height_dif = buffer.shape[1] - crop_size
        height_index = np.random.randint(height_dif) if (height_dif > 0) else 0
        width_dif = buffer.shape[2] - crop_size
        width_index = np.random.randint(width_dif) if (width_dif > 0) else 0

        # crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        # buffer = buffer[time_index:time_index + clip_len, :, :, :]

        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer



    def __len__(self):

        return len(self.file_list)

    def deal_with():
        pass

if __name__ == '__main__':
    train_dataloader = \
        DataLoader(
            Image_Dataset(params['train_dataset'], action='train', clip_len=params['clip_len'],
                          frame_sample_rate=params['frame_sample_rate']),
            batch_size=1, shuffle=True, num_workers=params['num_workers'])
    for data,label in train_dataloader:
        print(data)