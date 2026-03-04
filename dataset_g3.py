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


class Train_CoData(data.Dataset):
    def __init__(self, dut_img_root, dut_label_root, coco_img_root, coco_label_root, image_size, max_num):
        self.size_train = image_size   # 224*224
        self.data_size = (self.size_train, self.size_train)

        dut_class_list = os.listdir(dut_img_root)   #[class1,class2,...]
        self.dut_image_dirs = list(map(lambda x: os.path.join(dut_img_root, x), dut_class_list))  #[class1_path,class2_path,...]
        self.dut_label_dirs = list(map(lambda x: os.path.join(dut_label_root, x), dut_class_list))

        coco_class_list = os.listdir(coco_img_root)   #[class1,class2,...]
        self.coco_image_dirs = list(map(lambda x: os.path.join(coco_img_root, x), coco_class_list))  #[class1_path,class2_path,...]
        self.coco_label_dirs = list(map(lambda x: os.path.join(coco_label_root, x), coco_class_list))

        self.image_dirs = self.dut_image_dirs + self.coco_image_dirs
        self.label_dirs = self.dut_label_dirs + self.coco_label_dirs

        inds = [i for i in range(len(self.image_dirs))] # [0, 1, 2...,355]
        np.random.shuffle(inds)

        self.image_dirs = [self.image_dirs[i] for i in inds]
        self.label_dirs = [self.label_dirs[i] for i in inds]

        self.max_num = max_num   #Batch_size
        self.is_train = True
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.load_all = False

    def __getitem__(self, item):
        names = os.listdir(self.image_dirs[item]) #[class_x_img_name1, class_x_img_name2,...]
        num = len(names)  #class_1=14
        image_paths = list(map(lambda x: os.path.join(self.image_dirs[item], x), names)) #[class_1_img1_path,class_1_img2_path,...]
        label_paths = list(map(lambda x: os.path.join(self.label_dirs[item], x[:-4]+'.png'), names))

        if self.is_train:
            # random pick one category
            other_cls_ls = list(range(len(self.image_dirs)))  # [0,1,2...290]
            other_cls_ls.remove(item)
            other_item1, other_item2 = random.sample(set(other_cls_ls), 2)

            other_names1 = os.listdir(self.image_dirs[other_item1])  #[class_x_img_name1, class_x_img_name2,...]
            other_num1 = len(other_names1)    #class_5=18
            other_image_paths_1 = list(map(lambda x: os.path.join(self.image_dirs[other_item1], x), other_names1))  #[class_5_img1_path,class_5_img2_path,...]
            other_label_paths_1 = list(map(lambda x: os.path.join(self.label_dirs[other_item1], x[:-4]+'.png'), other_names1))

            other_names2 = os.listdir(self.image_dirs[other_item2])  #[class_x_img_name1, class_x_img_name2,...]
            other_num2 = len(other_names2)    #class_5=18
            other_image_paths_2 = list(map(lambda x: os.path.join(self.image_dirs[other_item2], x), other_names2))  #[class_5_img1_path,class_5_img2_path,...]
            other_label_paths_2 = list(map(lambda x: os.path.join(self.label_dirs[other_item2], x[:-4]+'.png'), other_names2))

            final_num = min(num, other_num1, other_num2, self.max_num) # 14, 18, batch_size=8

            sampled_list = random.sample(range(num), final_num)  # [0,1,2...13], 8
            new_image_paths = [image_paths[i] for i in sampled_list]
            new_label_paths = [label_paths[i] for i in sampled_list]

            other_sampled_list_1 = random.sample(range(other_num1), final_num)
            new_other_image_paths_1 = [other_image_paths_1[i] for i in other_sampled_list_1]
            new_other_label_paths_1 = [other_label_paths_1[i] for i in other_sampled_list_1]

            other_sampled_list_2 = random.sample(range(other_num2), final_num)
            new_other_image_paths_2 = [other_image_paths_2[i] for i in other_sampled_list_2]
            new_other_label_paths_2 = [other_label_paths_2[i] for i in other_sampled_list_2]

            ##########
            new_image_paths = new_image_paths + new_other_image_paths_1 + new_other_image_paths_2
            image_paths = new_image_paths
            new_label_paths = new_label_paths + new_other_label_paths_1 + new_other_label_paths_2
            label_paths = new_label_paths

            final_num = final_num * 3

        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        labels = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):  # final_num * 3
            if self.load_all:
                # TODO
                image = self.images_loaded[idx]
                label = self.labels_loaded[idx]
            else:
                if not os.path.exists(image_paths[idx]):
                    image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else image_paths[idx].replace('.png', '.jpg')
                image = Image.open(image_paths[idx]).convert('RGB')
                if not os.path.exists(label_paths[idx]):
                    label_paths[idx] = label_paths[idx].replace('.jpg', '.png') if label_paths[idx][-4:] == '.jpg' else label_paths[idx].replace('.png', '.jpg')
                label = Image.open(label_paths[idx]).convert('L')

            subpaths.append(os.path.join(image_paths[idx].split(os.sep)[-2], image_paths[idx].split(os.sep)[-1][:-4]+'.png'))  #[1/img_name.png, ]
            ori_sizes.append((image.size[1], image.size[0]))

            # loading image and label
            if self.is_train:
                image, label = cv_random_flip(image, label)
                image, label = random_crop(image, label)
                image, label = random_rotate(image, label)
                image = color_enhance(image)
                label = random_pepper(label)

            image, label = self.transform_image(image), self.transform_label(label)

            images[idx] = image
            labels[idx] = label

        cls_ls = [item] * (final_num // 3) + [other_item1] * (final_num // 3) + [other_item2] * (final_num // 3)
        return images, labels, subpaths, ori_sizes, cls_ls

    def __len__(self):
        return len(self.image_dirs)

def Train_get_loader(dut_img_root, dut_gt_root, coco_img_root, coco_gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, pin=False):
    dataset = Train_CoData(dut_img_root, dut_gt_root, coco_img_root, coco_gt_root, img_size, max_num)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader

##################################################
class Test_CoData(data.Dataset):
    def __init__(self, test_img_root, test_gt_root,  image_size, max_num):
        self.size_test = image_size   # 224*224
        self.data_size = (self.size_test, self.size_test)

        dut_class_list = os.listdir(test_img_root)   #[class1,class2,...]
        self.dut_image_dirs = list(map(lambda x: os.path.join(test_img_root, x), dut_class_list))  #[class1_path,class2_path,...]
        self.dut_label_dirs = list(map(lambda x: os.path.join(test_gt_root, x), dut_class_list))

        self.image_dirs = self.dut_image_dirs
        self.label_dirs = self.dut_label_dirs

        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.load_all = False

    def __getitem__(self, item):
        names = os.listdir(self.image_dirs[item]) #[class_x_img_name1, class_x_img_name2,...]
        num = len(names)  #class_1=14
        image_paths = list(map(lambda x: os.path.join(self.image_dirs[item], x), names)) #[class_1_img1_path,class_1_img2_path,...]
        label_paths = list(map(lambda x: os.path.join(self.label_dirs[item], x[:-4]+'.png'), names))
        final_num = num
        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        labels = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):  # final_num * 3
            if self.load_all:
                # TODO
                image = self.images_loaded[idx]
                label = self.labels_loaded[idx]
            else:
                if not os.path.exists(image_paths[idx]):
                    image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else image_paths[idx].replace('.png', '.jpg')
                image = Image.open(image_paths[idx]).convert('RGB')
                if not os.path.exists(label_paths[idx]):
                    label_paths[idx] = label_paths[idx].replace('.jpg', '.png') if label_paths[idx][-4:] == '.jpg' else label_paths[idx].replace('.png', '.jpg')
                label = Image.open(label_paths[idx]).convert('L')

            subpaths.append(os.path.join(image_paths[idx].split(os.sep)[-2], image_paths[idx].split(os.sep)[-1][:-4]+'.png'))  #[1/img_name.png, ]
            ori_sizes.append((image.size[1], image.size[0]))

            image, label = self.transform_image(image), self.transform_label(label)

            images[idx] = image
            labels[idx] = label

        return images, labels, subpaths, ori_sizes

    def __len__(self):
        return len(self.image_dirs)

def Test_get_loader(test_img_root, test_gt_root, img_size, batch_size, max_num = float('inf'), shuffle=False, num_workers=0, pin=False):
    dataset = Test_CoData(test_img_root, test_gt_root, img_size, max_num)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
