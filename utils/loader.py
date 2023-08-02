import math
import os
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from skimage import img_as_float32 as img_as_float
import cv2
import torch.nn.functional as F

def read_img(filename):
	img = cv2.imread(filename)
	img = img[:,:,::-1] / 255.0
	img = np.array(img).astype('float32')
	return img


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).astype('float32')


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')


def get_patch(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W - ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H - ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy + ps_temp, xx:xx + ps_temp, :]

    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i])
    return imgs




class DatasetFromFolder_train(Dataset):
    def __init__(self, image_dir, patch_size):
        super(DatasetFromFolder_train, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.patch_size = patch_size
        self.repeat = 80

    def __getitem__(self, index):
        index = self._get_index(index)
        target = read_img(self.image_filenames[index])
        [target] = get_patch([target], self.patch_size)
        return hwc_to_chw(target)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.image_filenames)


class DatasetFromFolder_test(Dataset):
    def __init__(self, noisy_dir, eval=False, save_img=False):
        super(DatasetFromFolder_test, self).__init__()
        self.noisy_filenames = [os.path.join(noisy_dir, x) for x in os.listdir(noisy_dir)]
        self.eval = eval
        self.save_img = save_img

    def __getitem__(self, index):
        noisy_name = self.noisy_filenames[index]
        noisy = read_img(noisy_name)
        _, file_name = os.path.split(self.noisy_filenames[index])

        if self.eval:
            target_name = noisy_name.replace('SIDD_noisy', 'SIDD_clean')        
            target = read_img(target_name)
            
            if self.save_img:
                return hwc_to_chw(noisy), hwc_to_chw(target), file_name[:-4]
            else:
                return hwc_to_chw(noisy), hwc_to_chw(target)
        
        else:
            if self.save_img:
                return hwc_to_chw(noisy), file_name[:-4]
            else:
                return hwc_to_chw(noisy)

    def __len__(self):
        return len(self.noisy_filenames)
