import os
import wandb
import torch
import random
import argparse
from rich.console import Console
from model import *
from trainer import *


def main(opt):
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    '''logger'''
    # logger = get_logger(f'./asset/log/{opt.model}.log')
    # logger.info(opt)

    console = Console(color_system='256', style=None)
    if opt.wandb:
        wandb.init(project="dr_diffuse")
        # wandb.init(project="dr_diffuse", settings=wandb.Settings(start_method="fork"))
    else:
        print(console.print("wandb forbidden!"))

    '''load data'''
    tr_data = VBDataset(
        './data/voicebank/noisy_trainset_wav',
        './data/voicebank/clean_trainset_wav',
        'train',
        opt)
    cv_data = VBDataset(
        './data/voicebank/noisy_testset_wav',
        './data/voicebank/clean_testset_wav',
        'valid',
        opt)

    console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')

    '''load model'''
    model = eval(opt.model)()

    '''load trainer'''
    trainer = VBTrainer(tr_data, cv_data, model, console, opt)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2023, help='manual seed')

    parser.add_argument('--model', type=str, default="Base", help='Base/DiffuSE')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="weight decay")
    parser.add_argument('--half_lr', type=int, default=3, help='decay learning rate to half scale')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop training')

    parser.add_argument('--win_size', type=int, default=320)
    parser.add_argument('--fft_num', type=int, default=320)
    parser.add_argument('--win_shift', type=int, default=160)
    parser.add_argument('--chunk_length', type=int, default=48000)
    parser.add_argument('--feat_type', type=str, default='sqrt', help='normal/sqrt/cubic/log_1x')

    parser.add_argument('--wandb', action='store_true', help='load wandb or not')

    args = parser.parse_args()
    args.device = torch.device('cuda:0')

    print(f'workspace:{os.getcwd()}, training device:{args.device}')

    main(args)
