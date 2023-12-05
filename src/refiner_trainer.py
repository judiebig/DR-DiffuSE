from abc import abstractmethod

import torch
import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from rich.progress import Progress
from model import *
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')


class BasicTrainer:
    def __init__(self, train_data, valid_data, console, opt):
        # dataset
        self.train_data = train_data,
        self.valid_data = valid_data

        collate = CustomCollate(opt)

        # data
        self.train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        self.valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)

        # c_gen
        self.c_gen = Base()
        checkpoint = torch.load("./asset/selected_model/c_gen.pth")
        self.c_gen.load_state_dict(checkpoint['model_state_dict'])
        self.c_gen.to(opt.device)
        self.c_gen.eval()

        # ddpm
        self.ddpm_model = DiffuSE(opt.params)
        checkpoint = torch.load(f"./asset/selected_model/ddpm.pth")
        self.ddpm_model.load_state_dict(checkpoint['model_state_dict'])
        self.ddpm_model.to(opt.device)
        self.ddpm_model.eval()

        # refiner
        self.refiner = Base().to(opt.device)
        if opt.from_base:
            checkpoint = torch.load("./asset/selected_model/c_gen.pth")
            self.refiner.load_state_dict(checkpoint['model_state_dict'])
            self.refiner.to(opt.device)

        # optimizer
        self.optim = torch.optim.Adam(self.refiner.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # others
        self.opt = opt
        self.console = console
        self.progress = None

    @abstractmethod
    def run_step(self, x):
        pass

    @abstractmethod
    def train(self):
        pass

    def save_cpt(self, step, save_path):
        torch.save(
            {
                'step': step,
                'model_state_dict': self.refiner.state_dict(),
                'optimizer_state_dict': self.refiner.state_dict()
            },
            save_path
        )


class RefinerTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, console, opt):
        super(RefinerTrainer, self).__init__(train_data, valid_data, console, opt)

        self.params = opt.params
        beta = np.array(self.params.noise_schedule)  # noise_schedule --> beta
        noise_level = np.cumprod(1 - beta)  # noise_level --> alpha^bar
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(self.opt.device)
        self.alpha, self.beta, self.alpha_cum, self.sigmas, self.T = self.inference_schedule(
            fast_sampling=self.params.fast_sampling)

    def inference_schedule(self, fast_sampling=False):
        """
        Compute fixed parameters in ddpm

        :return:
            alpha:          alpha for training,         size like noise_schedule
            beta:           beta for inference,         size like inference_noise_schedule or noise_schedule
            alpha_cum:      alpha_cum for inference
            sigmas:         sqrt(beta_t^tilde)
            T:              Timesteps
        """
        training_noise_schedule = np.array(self.params.noise_schedule)
        inference_noise_schedule = np.array(
            self.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule  # alpha_t for train
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        sigmas = [0 for i in alpha]
        for n in range(len(alpha) - 1, -1, -1):
            sigmas[n] = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5  # sqrt(beta_t^tilde)
        # print("sigmas", sigmas)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)  # from schedule to T which as the input of model
                    break
        T = np.array(T, dtype=np.float32)
        return alpha, beta, alpha_cum, sigmas, T

    def data_compress(self, x):
        batch_feat = x['feats']
        batch_label = x['labels']
        noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
        clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])
        if self.opt.feat_type == 'normal':
            batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
        elif self.opt.feat_type == 'sqrt':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                torch.norm(batch_label, dim=1)) ** 0.5
        elif self.opt.feat_type == 'cubic':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                torch.norm(batch_label, dim=1)) ** 0.3
        elif self.opt.feat_type == 'log_1x':
            batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                torch.log(torch.norm(batch_label, dim=1) + 1)
        if self.opt.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
            batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                     dim=1)
            batch_label = torch.stack((batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                                      dim=1)
        return batch_feat, batch_label

    def run_step(self, x):
        """
        (1) load stft version from data_loader;
        (2) compress;
        (3) generate c via c_gen;
        (4) feed c to ddpm and generate augmented data;
        (5) feed generated data to refiner
        (6) train refiner
        :param x:
        :return:
        """
        with torch.no_grad():
            batch_feat, batch_label = self.data_compress(x)  # (2)
            N = batch_label.shape[0]  # batch size

            c = self.c_gen(batch_feat)['est_comp']  # (3)

            # (4)
            noise_scale = self.alpha_cum[-1]
            noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
            noise = torch.randn_like(batch_feat)  # epsilon           [N, 2, T, F]
            x_T = noise_scale_sqrt * (c) + (1.0 - noise_scale) ** 0.5 * noise
            spec = x_T

            for n in range(len(self.alpha) - 1, -1, -1):

                t = torch.tensor([self.T[n]], device=spec.device).repeat(N)
                out = self.ddpm_model(spec, c, t)

                c1 = 1 / self.alpha[n] ** 0.5  # for ddpm sampling
                c2 = self.beta[n] / (1 - self.alpha_cum[n]) ** 0.5  # for ddpm sampling
                spec = c1 * (spec - c2 * out['est_noise'])
                if n > 0:  # + random noise
                    noise = torch.randn_like(spec)
                    spec += self.sigmas[n] * noise

                    # add condition guidance
                    noise_scale = torch.Tensor([self.alpha_cum[n]]).unsqueeze(1).unsqueeze(2).unsqueeze(
                        3).cuda()  # alpha_bar_t [N, 1, 1, 1]
                    noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
                    noise = torch.randn_like(c).cuda()  # epsilon           [N, 2, T, F]
                    c_t = noise_scale_sqrt * c + (1.0 - noise_scale) ** 0.5 * noise  # c_t
                    spec = 0.5 * spec + 0.5 * c_t
        spec = 0.5 * spec + 0.5 * batch_feat
        out = self.refiner(spec)
        loss = com_mag_mse_loss(out['est_comp'], batch_label, x['frame_num_list'])
        return {
            'model_out': out,
            'loss': loss,
            'compressed_feats': batch_feat,
            'compressed_label': batch_label
        }

    def train(self):
        prev_cv_loss = float("inf")
        best_cv_loss = float("inf")
        cv_no_impv = 0
        harving = False


        with Progress() as self.progress:
            for epoch in range(self.opt.n_epoch):
                batch_train = self.progress.add_task(f"[green]training epoch_{epoch}...", total=len(self.train_loader))
                self.refiner.train()
                for batch in self.train_loader:
                    # cuda
                    for key in batch.keys():
                        try:
                            batch[key] = batch[key].to(self.opt.device)
                        except AttributeError:
                            continue
                    out = self.run_step(batch)
                    self.optim.zero_grad()
                    out['loss'].backward()
                    self.optim.step()
                    self.progress.advance(batch_train, advance=1)
                    if self.opt.wandb:
                        wandb.log(
                            {
                                'train_loss': out['loss'].item()
                            }
                        )

                mean_valid_loss = self.inference()

                '''Adjust the learning rate and early stop'''
                if self.opt.half_lr > 1:
                    if mean_valid_loss >= prev_cv_loss:
                        cv_no_impv += 1
                        if cv_no_impv == self.opt.half_lr:
                            harving = True
                        if cv_no_impv >= self.opt.early_stop > 0:
                            print("No improvement and apply early stop")
                            return
                    else:
                        cv_no_impv = 0

                if harving == True:
                    optim_state = self.optim.state_dict()
                    for i in range(len(optim_state['param_groups'])):
                        optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                    self.optim.load_state_dict(optim_state)
                    print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                    harving = False
                prev_cv_loss = mean_valid_loss

                if mean_valid_loss < best_cv_loss:
                    print(
                        f"best loss is: {best_cv_loss}, current loss is: {mean_valid_loss}, save best_checkpoint.pth")
                    best_cv_loss = mean_valid_loss

                    self.save_cpt(epoch,
                                  save_path=f'./asset/model/'
                                            f'refiner_{self.refiner.__class__.__name__}_best.pth')
                self.save_cpt(epoch,
                              save_path=f'./asset/model/'
                                        f'refiner_{self.refiner.__class__.__name__}_{epoch}.pth')

    @torch.no_grad()
    def inference(self):
        self.refiner.eval()
        if self.progress:
            loss = self.inference_()
        else:
            with Progress() as self.progress:
                loss = self.inference_()
        return loss

    def inference_(self):
        loss_list = []
        csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = [], [], [], [], [], []
        batch_valid = self.progress.add_task(f"[green]validating...", total=len(self.valid_loader))
        for batch in self.valid_loader:
            # cuda
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.opt.device)
                except AttributeError:
                    continue
            out = self.run_step(batch)  # out['compressed_feats']
            batch_result = compare_complex(out['model_out']['est_comp'], out['compressed_label'],
                                           batch['frame_num_list'],
                                           feat_type=self.opt.feat_type)

            loss_list.append(out['loss'].item())
            csig_list.append(batch_result[0])
            cbak_list.append(batch_result[1])
            covl_list.append(batch_result[2])
            pesq_list.append(batch_result[3])
            ssnr_list.append(batch_result[4])
            stoi_list.append(batch_result[5])

            self.progress.advance(batch_valid, advance=1)

        if self.opt.wandb:
            wandb.log(
                {
                    'test_loss': np.mean(loss_list),
                    'test_mean_csig': np.mean(csig_list),
                    'test_mean_cbak': np.mean(cbak_list),
                    'test_mean_covl': np.mean(covl_list),
                    'test_mean_pesq': np.mean(pesq_list),
                    'test_mean_ssnr': np.mean(ssnr_list),
                    'test_mean_stoi': np.mean(stoi_list),
                }
            )
        else:
            print({
                'test_loss': np.mean(loss_list),
                'test_mean_csig': np.mean(csig_list),
                'test_mean_cbak': np.mean(cbak_list),
                'test_mean_covl': np.mean(covl_list),
                'test_mean_pesq': np.mean(pesq_list),
                'test_mean_ssnr': np.mean(ssnr_list),
                'test_mean_stoi': np.mean(stoi_list),
            })

        return np.mean(loss_list)

    def save_cpt(self, step, save_path):
        super().save_cpt(step, save_path)
