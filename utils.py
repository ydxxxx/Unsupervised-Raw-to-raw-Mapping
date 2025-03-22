import torch
import torch.nn.functional as F
import logging
from torch import pixel_shuffle, pixel_unshuffle
import cv2
import numpy as np
import os


def calculate_psnr(pre_image, image):
    sum_PSNR = 0
    for i in range(len(image)):
        mse = torch.mean((pre_image[i] - image[i]) ** 2).to('cpu')
        psnr = 10*torch.log10(1/mse)
        sum_PSNR += psnr.detach().data.item()
    return sum_PSNR / len(image)


def pad_image(image, patch_size):
    h, w = image.shape[-2:]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    image = F.pad(image, (0, pad_w, 0, pad_h))
    return image

def split_large_image(**kwargs):
    image, type = kwargs['image'],kwargs['type']
    if type=='split':
        size = kwargs['patch_size']
        height, width = image.shape[:2]
        # 计算分割后的小图片数量
        num_cols = width // size
        num_rows = height // size
        small_images = []
        for row in range(num_rows):
            for col in range(num_cols):
                # 计算每个小图片的位置
                left = col * size
                top = row * size
                right = left + size
                bottom = top + size
                # 分割并保存小图片
                small_image = image[top:bottom,left:right, ...]
                small_images.append(small_image)
        return small_images
    elif type=='zoom':
        processed_image_size = kwargs['processed_image_size']
        return [cv2.resize(image,processed_image_size)]


def space_to_depth(raw):
    if not torch.is_tensor(raw):
        raw = torch.tensor(raw)
    raw = pixel_unshuffle(raw, 2)
    raw = np.array(raw)
    return raw


def depth_to_space(feature_maps):
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.tensor(feature_maps)
    feature_maps = pixel_shuffle(feature_maps, 2)
    feature_maps = np.array(feature_maps.to('cpu'))
    return feature_maps



class Logger:
    #######################################################################################################################################
    ######################  Code based on https://github.com/sanechips-multimedia/syenet/logger.py ########################################
    #######################################################################################################################################
    def __init__(self,args):
        self.args = args
        self.log_path = os.path.join(args.save_path, 'logger.log')

        self.logging_level = logging.DEBUG
        self.file_level = logging.DEBUG
        self.stream_level = logging.DEBUG

        self.logger = logging.getLogger('logger.log')
        self.logger.setLevel(self.logging_level)

        self.configure()

    def configure(self):
        log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(log_format)

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(log_format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
