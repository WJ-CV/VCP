import os
from PIL import Image, ImageEnhance
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random

from preproc import cv_random_flip, random_crop, random_rotate, color_enhance, random_gaussian, random_pepper


##################################################
class Test_CoData(data.Dataset):
    def __init__(self, test_img_root,  image_size, max_num):
        self.size_test = image_size   # 224*224
        self.data_size = (self.size_test, self.size_test)

        dut_class_list = os.listdir(test_img_root)   #[class1,class2,...]
        self.dut_image_dirs = list(map(lambda x: os.path.join(test_img_root, x), dut_class_list))  #[class1_path,class2_path,...]

        self.image_dirs = self.dut_image_dirs

        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.load_all = False

    def __getitem__(self, item):
        names = os.listdir(self.image_dirs[item]) #[class_x_img_name1, class_x_img_name2,...]
        num = len(names)  #class_1=14
        image_paths = list(map(lambda x: os.path.join(self.image_dirs[item], x), names)) #[class_1_img1_path,class_1_img2_path,...]
        final_num = num
        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):  # final_num * 3
            if self.load_all:
                # TODO
                image = self.images_loaded[idx]
            else:
                if not os.path.exists(image_paths[idx]):
                    image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else image_paths[idx].replace('.png', '.jpg')
                image = Image.open(image_paths[idx]).convert('RGB')

            subpaths.append(os.path.join(image_paths[idx].split(os.sep)[-2], image_paths[idx].split(os.sep)[-1][:-4]+'.png'))  #[1/img_name.png, ]
            ori_sizes.append((image.size[1], image.size[0]))

            image = self.transform_image(image)

            images[idx] = image

        return images, subpaths, ori_sizes

    def __len__(self):
        return len(self.image_dirs)

def Test_get_loader(test_img_root, img_size, batch_size, max_num = float('inf'), shuffle=False, num_workers=0, pin=False):
    dataset = Test_CoData(test_img_root, img_size, max_num)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
