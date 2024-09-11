import os
from generator import Conformer
from discriminator import Discriminator
import torch
import torch.nn as nn
import itertools
from pytorch_msssim import SSIM
import random
from data import Test_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_psnr, Logger

class Tester:
    def __init__(self, args):
        super(Tester, self).__init__(args)
        self.logger = Logger(args.save_path)
        self.args = args
        self.device = args.device
        self.mae = nn.L1Loss()
        self.ssim = SSIM(data_range=1, channel=args.in_channels)
        test_dataset = Test_dataset(args.device, args.test_data_path, args.camera1, args.camera2)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,shuffle=False,pin_memory=False)
        self.gen = Conformer(
            in_channels=self.args.in_channels,
            base_channel=self.args.base_channels,
            embed_dim=self.args.embed_dim,
            depth=self.args.depth,
            down_scale_times=self.args.down_scale_times
        ).to(self.device)
        self.logger = Logger(args)
        assert os.path.exists(args.weights_path)
        weights = torch.load(args.weights_path)
        self.gen.load_state_dict(weights)

    def test(self):
        self.logger.info(f'start test with weights {self.args.weights_path}')
        self.gen.eval()
        with torch.no_grad():
            total_mae = 0
            total_psnr = 0
            total_ssim = 0
            for origin, target in tqdm(self.test_dataloader):
                pre_target = self.gen(origin)
                total_mae += self.mae(pre_target, target)
                total_psnr += calculate_psnr(pre_target, target)
                total_ssim += self.ssim(pre_target, target)
            self.logger.info(f'mae: {total_mae/len(self.test_dataloader)}')
            self.logger.info(f'psnr: {total_psnr/len(self.test_dataloader)}')
            self.logger.info(f'ssim: {total_ssim/len(self.test_dataloader)}')
