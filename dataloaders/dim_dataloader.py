import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from srcs.models.config import im_size, unknown_code, num_valid
from srcs.models.utils import safe_crop

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('data/DIM/Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('data/DIM/Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('data/DIM/Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('data/DIM/Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(im_name, bg_name, alpha_name):
    im = cv.imread(im_name)
    a = cv.imread(alpha_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)


def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y


class DIMDataset(Dataset):
    def __init__(self, split, fg_dir, matte_dir, bg_dir, data_root="data/DIM/"):
        self.split = split
        self.fg_dir = fg_dir
        self.matter_dir = matte_dir
        self.bg_dir = bg_dir

        filename = os.path.join(data_root, '{}_names.txt'.format(split))
        #print("filename=", filename)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()
        
        print(split, "   ", len(self.names))

        #print(self.names)
        self.transformer = data_transforms[split]

        if split=='train':
            self.fg_files = fg_files
            self.bg_files = bg_files
        else:
            self.fg_files = fg_test_files
            self.bg_files = bg_test_files


    def __getitem__(self, i):
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        #print("fcount=", fcount, "   bcount=", bcount)
        im_name = os.path.join(self.fg_dir, self.fg_files[fcount])
        bg_name = os.path.join(self.bg_dir, self.bg_files[bcount])
        alpha_name = os.path.join(self.matter_dir, self.fg_files[fcount])
        img, alpha, fg, bg = process(im_name, bg_name, alpha_name)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)
        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        trimap = gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        alpha_downsized =  cv.resize(alpha, (int(im_size / 16), int(im_size/ 16)))
        #print("alpha.shape=", alpha.shape)
        #down_lbl = cv2.blur(down_lbl, (3, 3))
        alpha_downsized = cv.GaussianBlur(alpha_downsized,(3,3), 0)
        #alpha_downsized = alpha_downsized[:, :, np.newaxis]

        #print("alpha_downsized.shape=", alpha_downsized.shape)

        #x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        #x[0:3, :, :] = img
        #x[3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        label = torch.from_numpy(alpha/255.)
        label_down = torch.from_numpy(alpha_downsized/255.)
        '''
        y = np.empty((2, im_size, im_size), dtype=np.float32)
        y[0, :, :] = torch.from_numpy(alpha / 255.)
        '''
        mask = np.equal(trimap, 128).astype(np.float32)
        mask = torch.from_numpy(mask)
        #y[1, :, :] = mask
        trimap = torch.from_numpy(np.ascontiguousarray(trimap))
        #print("max=", torch.max(trimap), "  mask=", mask, "  label=", label[0])
        return img, label, label_down, mask, trimap

    def __len__(self):
        return len(self.names)


def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == "__main__":
    gen_names()