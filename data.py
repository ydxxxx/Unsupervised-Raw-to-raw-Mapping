from torch.utils.data import Dataset
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
import random
import scipy.io
from utils import pad_image
import matplotlib.pyplot as plt

class RandRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        random_angle = self.degrees[torch.randint(low=0,high=len(self.degrees),size=[1])]
        return rotate(img, random_angle)

class Train_dataset(Dataset):
    def __init__(self, device, data_path, origin_camera, target_camera,enable_data_augmentation):
        super(Train_dataset,self).__init__()
        self.device = device
        self.origin_camera = origin_camera
        self.target_camera = target_camera
        self.origin_camera_dir = os.path.join(data_path, origin_camera)
        self.target_camera_dir = os.path.join(data_path, target_camera)

        self.origin_raws = os.listdir(self.origin_camera_dir)
        self.target_raws = os.listdir(self.target_camera_dir)
        self.origin_len = len(self.origin_raws)
        self.target_len = len(self.target_raws)
        self.dataset_size = min(len(self.origin_raws),len(self.target_raws))
        self.origin_raws = sorted(self.origin_raws)
        self.target_raws = sorted(self.target_raws)
        if enable_data_augmentation:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                RandRotation([0, 90, 180, 270]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        seed = torch.random.seed()
        origin_idx = idx
        target_idx = idx

        origin_raw_path = os.path.join(self.origin_camera_dir, self.origin_raws[origin_idx])
        origin_raw = torch.tensor(np.load(origin_raw_path))
        torch.random.manual_seed(seed)
        origin_raw = self.transform(origin_raw).to(self.device)

        target_raw_path = os.path.join(self.target_camera_dir, self.target_raws[target_idx])
        target_raw = torch.tensor(np.load(target_raw_path))
        torch.random.manual_seed(seed)
        target_raw = self.transform(target_raw).to(self.device)
        return origin_raw, target_raw

class Test_dataset(Dataset):
    def __init__(self,device,data_path,origin_camera,target_camera):
        self.device = device
        suffix = 'raw-rggb'
        self.origin_camera_dir = os.path.join(data_path,origin_camera,suffix)
        self.target_camera_dir = os.path.join(data_path,target_camera,suffix)
        self.dataset_size = min(len(os.listdir(self.origin_camera_dir)),len(os.listdir(self.target_camera_dir)))
        self.origin_raws = sorted(os.listdir(self.origin_camera_dir))
        self.target_raws = sorted(os.listdir(self.target_camera_dir))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        origin_raw_path = os.path.join(self.origin_camera_dir, self.origin_raws[idx])
        origin_raw = torch.tensor(scipy.io.loadmat(origin_raw_path)['raw_rggb'],dtype=torch.float).permute(2,1,0)
        origin_raw = self.transform(origin_raw).to(self.device)

        target_raw_path = os.path.join(self.target_camera_dir, self.target_raws[idx])
        target_raw = torch.tensor(scipy.io.loadmat(target_raw_path)['raw_rggb'],dtype=torch.float).permute(2,1,0)
        target_raw = self.transform(target_raw).to(self.device)
        return origin_raw, target_raw
