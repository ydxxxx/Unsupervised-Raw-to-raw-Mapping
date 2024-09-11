import os
from generator import Conformer
from discriminator import Discriminator
import torch
import torch.nn as nn
import itertools
from pytorch_msssim import SSIM
import random
from data import Train_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_psnr, Logger


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.args = args
        self.logger = Logger(args)
        self.logger.info('Start training with args:')
        self.logger.info('\n'.join(f'{key}: {value}' for key, value in vars(self.args).items()))

        self.device = args.device
        self._initialize_models()
        self._initialize_optimizers()
        self.train_state = self.load_ckpt()
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.mae = nn.L1Loss()
        self.ssim = SSIM(data_range=1, channel=args.in_channels)
        self.target_real = torch.ones([args.train_batch_size, 1], requires_grad=False).to(args.device)
        self.target_fake = torch.zeros([args.train_batch_size, 1], requires_grad=False).to(args.device)
        self.fake_a_buffer = ReplayBuffer()
        self.fake_b_buffer = ReplayBuffer()
        train_dataset = Train_dataset(args.device, args.train_data_path, args.camera1, args.camera2,enable_data_augmentation=True)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                                           shuffle=True,pin_memory=False, drop_last=True)
        val_dataset = Train_dataset(args.device, args.test_data_path, args.camera1, args.camera2,enable_data_augmentation=False)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size,
                                         shuffle=False,pin_memory=False, drop_last=True)

    def _initialize_models(self):
        self.gen_a2b = Conformer(
            in_channels=self.args.in_channels,
            base_channel=self.args.base_channels,
            embed_dim=self.args.embed_dim,
            depth=self.args.depth,
            down_scale_times=self.args.down_scale_times
        ).to(self.device)

        self.gen_b2a = Conformer(
            in_channels=self.args.in_channels,
            base_channel=self.args.base_channels,
            embed_dim=self.args.embed_dim,
            depth=self.args.depth,
            down_scale_times=self.args.down_scale_times,
            share_model=self.gen_a2b
        ).to(self.device)
        self.dis_a = Discriminator(self.args.in_channels).to(self.device)
        self.dis_b = Discriminator(self.args.in_channels).to(self.device)

    def _initialize_optimizers(self):
        self.optimizer_gen = torch.optim.Adam(
            itertools.chain(self.gen_a2b.parameters(), self.gen_b2a.parameters()),
            lr=self.args.gen_lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_dis_a = torch.optim.Adam(
            self.dis_a.parameters(),
            lr=self.args.dis_lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_dis_b = torch.optim.Adam(
            self.dis_b.parameters(),
            lr=self.args.dis_lr,
            betas=(0.5, 0.999)
        )

    def _initialize_train_state(self):
        return {
            'epoch': 0,
            'gen_a2b_state_dict': self.gen_a2b.state_dict(),
            'gen_b2a_state_dict': self.gen_b2a.state_dict(),
            'dis_a_state_dict': self.dis_a.state_dict(),
            'dis_b_state_dict': self.dis_b.state_dict(),
            'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
            'optimizer_dis_a_state_dict': self.optimizer_dis_a.state_dict(),
            'optimizer_dis_b_state_dict': self.optimizer_dis_b.state_dict(),
            'gen_losses': [],
            'gen_cycle_losses': [],
            'gen_gan_losses': [],
            'gen_identity_losses': [],
            'dis_losses': [],
            'a2b_mae': [],
            'b2a_mae': [],
            'a2b_psnr': [],
            'b2a_psnr': [],
            'a2b_ssim': [],
            'b2a_ssim': []
        }

    def load_ckpt(self):
        if os.path.exists(self.args.ckpt_path):
            self.logger.info(f'load ckpt from{self.args.ckpt_path}')
            ckpt = torch.load(self.args.ckpt_path)
            self.gen_a2b.load_state_dict(ckpt['gen_a2b_state_dict'])
            self.gen_b2a.load_state_dict(ckpt['gen_b2a_state_dict'])
            self.dis_a.load_state_dict(ckpt['dis_a_state_dict'])
            self.dis_b.load_state_dict(ckpt['dis_b_state_dict'])
            self.optimizer_gen.load_state_dict(ckpt['optimizer_gen_state_dict'])
            self.optimizer_dis_a.load_state_dict(ckpt['optimizer_dis_a_state_dict'])
            self.optimizer_dis_b.load_state_dict(ckpt['optimizer_dis_b_state_dict'])
            return ckpt
        else:
            self.logger.info(f'init a new ckpt')
            return self._initialize_train_state()

    def save_ckpt(self):
        self.train_state['gen_a2b_state_dict'] = self.gen_a2b.state_dict()
        self.train_state['gen_b2a_state_dict'] = self.gen_b2a.state_dict()
        self.train_state['dis_a_state_dict'] = self.dis_a.state_dict()
        self.train_state['dis_b_state_dict'] = self.dis_b.state_dict()
        self.train_state['optimizer_gen_state_dict'] = self.optimizer_gen.state_dict()
        self.train_state['optimizer_dis_a_state_dict'] = self.optimizer_dis_a.state_dict()
        self.train_state['optimizer_dis_b_state_dict'] = self.optimizer_dis_b.state_dict()
        epoch = self.train_state['epoch']
        torch.save(self.train_state, os.path.join(self.args.save_path, f'train_state_{epoch}.pth'))
        if self.train_state['a2b_ssim'][-1] + self.train_state['b2a_ssim'][-1] == max([self.train_state['a2b_ssim'][i]+self.train_state['b2a_ssim'][i] for i in range(len(self.train_state['a2b_ssim']))]):
            torch.save(self.train_state['gen_a2b_state_dict'], os.path.join(self.args.save_path, f'gen_a2b_best.pth'))
            torch.save(self.train_state['gen_b2a_state_dict'], os.path.join(self.args.save_path, f'gen_b2a_best.pth'))

    def train(self):
        while self.train_state['epoch'] < self.args.epochs:
            epoch = self.train_state['epoch']
            self.logger.info(f'start training of epoch {epoch}')
            for key in self.train_state.keys():
                if 'losses' in key:
                    self.train_state['key'].append(0)
            self.gen_a2b.train()
            self.gen_b2a.train()
            for real_a, real_b in tqdm(self.train_dataloader):
                scale = len(real_a)/len(self.train_dataloader)
                ### generator ###
                same_b = self.gen_a2b(real_b)
                loss_identity_b = self.criterion_identity(same_b, real_b)
                same_a = self.gen_b2a(real_a)
                loss_identity_a = self.criterion_identity(same_a, real_a)

                # gan loss
                fake_b = self.gen_a2b(real_a)
                pre_fake = self.dis_b(fake_b)
                loss_gan_a2b = self.criterion_gan(pre_fake, self.target_real)
                fake_a = self.gen_b2a(real_b)
                pre_fake = self.dis_a(fake_a)
                loss_gan_b2a = self.criterion_gan(pre_fake, self.target_real)

                # cycle loss
                recover_a = self.gen_b2a(fake_b)
                loss_cycle_aba = self.criterion_cycle(recover_a, real_a)
                recover_b = self.gen_a2b(fake_a)
                loss_cycle_bab = self.criterion_cycle(recover_b, real_b)

                total_loss_gen = ((loss_identity_a + loss_identity_b) * 5 +
                                  loss_gan_a2b + loss_gan_b2a + (loss_cycle_aba + loss_cycle_bab) * 10)
                self.optimizer_gen.zero_grad()
                total_loss_gen.backward()
                torch.nn.utils.clip_grad_norm_(self.gen_a2b.parameters(), self.args.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.gen_b2a.parameters(), self.args.grad_clip)
                self.optimizer_gen.step()

                ### discriminator ###
                # dis a
                pred_real = self.dis_a(real_a)
                loss_dis_real = self.criterion_gan(pred_real, self.target_real)

                fake_a = self.fake_a_buffer.push_and_pop(fake_a.detach())
                pre_fake = self.dis_a(fake_a)
                loss_dis_fake = self.criterion_gan(pre_fake, self.target_fake)

                loss_dis_a = (loss_dis_real + loss_dis_fake) * 0.5

                self.optimizer_dis_a.zero_grad()
                loss_dis_a.backward()
                torch.nn.utils.clip_grad_norm_(self.dis_a.parameters(), self.args.grad_clip)
                self.optimizer_dis_a.step()
                # dis b
                pred_real = self.dis_b(real_b)
                loss_dis_real = self.criterion_gan(pred_real, self.target_real)

                fake_b = self.fake_b_buffer.push_and_pop(fake_b.detach())
                pre_fake = self.dis_b(fake_b)
                loss_dis_fake = self.criterion_gan(pre_fake, self.target_fake)

                loss_dis_b = (loss_dis_real + loss_dis_fake) * 0.5

                self.optimizer_dis_b.zero_grad()
                loss_dis_b.backward()
                torch.nn.utils.clip_grad_norm_(self.dis_b.parameters(), self.args.grad_clip)
                self.optimizer_dis_b.step()
                self.train_state['dis_losses'][-1] += (loss_dis_a + loss_dis_b).detach().data.item() / scale
                self.train_state['gen_losses'][-1] += total_loss_gen.detach().data.item() / scale
                self.train_state['gen_gan_losses'][-1] += (loss_gan_a2b + loss_gan_b2a).detach().data.item() / scale
                self.train_state['gen_cycle_losses'][-1] += (loss_cycle_bab + loss_cycle_aba).detach().data.item() / scale
                self.train_state['gen_identity_losses'][-1] += (loss_identity_a + loss_identity_b).detach().data.item() / scale
            self.logger.info(f'end training of epoch {epoch} ')
            self.logger.info('\n'.join(f'{key}: {value[-1]}' for key, value in vars(self.train_state).items() if 'losses' in key))
            self.val_epoch()
            self.train_state['epoch'] += 1
            if self.train_state%self.args.save_freq == 0:
                self.save_ckpt()

    def val_epoch(self):
        self.logger.info(f'start valid')
        self.train_state['a2b_mae'].append(0)
        self.train_state['b2a_mae'].append(0)
        self.train_state['a2b_ssim'].append(0)
        self.train_state['b2a_ssim'].append(0)
        self.train_state['a2b_psnr'].append(0)
        self.train_state['b2a_psnr'].append(0)
        self.gen_a2b.eval()
        self.gen_b2a.eval()
        with torch.no_grad():
            for real_a, real_b in tqdm(self.val_dataloader):
                scale = len(real_a)/len(self.val_dataloader)
                fake_b = self.gen_a2b(real_a)
                fake_a = self.gen_b2a(real_b)
                self.train_state['a2b_mae'][-1] += self.mae(fake_b, real_b).detach().data.item() / scale
                self.train_state['b2a_mae'][-1] += self.mae(fake_a, real_a).detach().data.item() / scale
                self.train_state['a2b_ssim'][-1] += self.ssim(fake_b, real_b).detach().data.item() / scale
                self.train_state['b2a_ssim'][-1] += self.ssim(fake_a, real_a).detach().data.item() / scale
                self.train_state['a2b_psnr'][-1] += calculate_psnr(fake_b, real_b) / scale
                self.train_state['b2a_psnr'][-1] += calculate_psnr(fake_a, real_a) / scale
            self.logger.info('\n'.join(f'{key}: {value[-1]}' for key, value in vars(self.train_state).items() if key in ['mae','ssim','psnr']))
