import argparse
from rich.console import Console
from utils import *
from refiner_trainer import *


def main(opt):
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    console = Console(color_system='256', style=None)

    if opt.wandb:
        wandb.init(project="dr_diffuse_new")
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

    # cv_data = VBDataset(
    #     './data/chime4/noisy',
    #     './data/chime4/clean',
    #     'valid',
    #     opt)

    console.print(f'evaluation: total {cv_data.__len__()} eval data.')

    '''load model'''
    opt.params = AttrDict(
        ours=False,
        fast_sampling=opt.fast_sampling,
        noise_schedule=np.linspace(1e-4, 0.05, 200).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
    )

    '''load trainer'''
    trainer = RefinerTrainer(tr_data, cv_data, console, opt)
    if opt.inference:
        checkpoint = torch.load(f"./asset/selected_model/refiner.pth")
        trainer.refiner.load_state_dict(checkpoint['model_state_dict'])
        trainer.refiner.to(opt.device)
        trainer.inference()
    else:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2023, help='manual seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=2,
                        help='number of epoch, if start from base, then could choose a small value such as 2')
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

    parser.add_argument('--fast_sampling', action='store_true', help='')
    parser.add_argument('--inference', action='store_true', help='')
    parser.add_argument('--from_base', action='store_true', help='')

    args = parser.parse_args()
    args.device = torch.device('cuda:0')

    print(f'workspace:{os.getcwd()}, training device:{args.device}')

    main(args)
