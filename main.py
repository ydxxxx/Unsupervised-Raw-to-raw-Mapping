import argparse
import os.path
import json
import torch
from trainer import Trainer
from tester import Tester


def get_argument():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--parent_path', default='./')
    parser.add_argument('--camera1', default='iphone-x')
    parser.add_argument('--camera2', default='samsung-s9')
    parser.add_argument('--train_data_path',default='dataset/processed/unpaired/')
    parser.add_argument('--val_data_path', default='dataset/processed/train/')
    parser.add_argument('--test_data_path', default='dataset/paired')
    parser.add_argument('--save_path', type=str, default='in_paper/simple_cnn')
    parser.add_argument('--ckpt_path',type=str,default=None)
    # training settings
    parser.add_argument('--mode', choices=['train', 'test'],default='train')
    parser.add_argument('--seed', default=114514)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=40)
    parser.add_argument('--start_epoch', type=int, default=-1)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--grad_clip', default=2)
    parser.add_argument('--gen_lr', type=float, default=1e-5)
    parser.add_argument('--dis_lr', type=float, default=2e-5)
    # conformer
    parser.add_argument('--share_params', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--down_scale_times', type=int, default=1)
    args = parser.parse_args()

    args.train_data_path = os.path.join(args.parent_path, args.train_data_path)
    args.test_data_path = os.path.join(args.parent_path, args.test_data_path)
    args.val_data_path = os.path.join(args.parent_path, args.val_data_path)
    args.ckpt_path = os.path.join(args.parent_path, args.ckpt_path)
    args.save_path = os.path.join(args.parent_path, args.save_path)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == '__main__':
    args = get_argument()
    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()
    elif args.mode == 'test':
        tester = Tester(args)
        tester.test()


