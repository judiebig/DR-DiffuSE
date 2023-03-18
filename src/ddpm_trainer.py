from abc import abstractmethod

import torch
import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from rich.progress import Progress
from model import *

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


class BasicTrainer:
    def __init__(self, train_data, valid_data, model, console, logger, opt):
        # dataset
        self.train_data = train_data,
        self.valid_data = valid_data

        collate = CustomCollate(opt)

        # data
        self.train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        self.valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)

        # model
        self.model_ddpm = model.to(opt.device)

        # optimizer
        self.optim_ddpm = torch.optim.Adam(self.model_ddpm.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # others
        self.opt = opt
        self.console = console
        self.logger = logger  # can be wandb, logging, or rich.console
        self.progress = None

    @abstractmethod
    def run_step(self, x):
        pass

    @abstractmethod
    def train(self):
        pass

    def save_cpt(self, step, save_path):
        """
        save checkpoint, for inference/re-training
        :return:
        """
        torch.save(
            {
                'step': step,
                'model_state_dict': self.model_ddpm.state_dict(),
                'optimizer_state_dict': self.model_ddpm.state_dict()
            },
            save_path
        )


class VBDDPMTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, console, logger, opt):
        super(VBDDPMTrainer, self).__init__(train_data, valid_data, model, console, logger, opt)

        self.params = opt.params
        beta = np.array(self.params.noise_schedule)  # noise_schedule --> beta
        noise_level = np.cumprod(1 - beta)  # noise_level --> alpha^bar
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(self.opt.device)

        # load conditon generator
        if opt.c_gen:
            self.c_gen = Base()
            checkpoint = torch.load("./asset/selected_model/base_model_pesq_312.pth")
            self.c_gen.load_state_dict(checkpoint['model_state_dict'])
            self.c_gen.to(opt.device)

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
    
    def data_reconstuct(self, x_complex, feat_type='sqrt'):
        x_mag, x_phase = torch.norm(x_complex, dim=1), torch.atan2(x_complex[:, -1, :, :], x_complex[:, 0, :, :])


        if self.opt.feat_type == 'sqrt':
            x_mag = x_mag ** 2
            x_com = torch.stack((x_mag * torch.cos(x_phase), x_mag * torch.sin(x_phase)), dim=1)
        else:
            pass
            # unfinished 

        return x_com

    def run_step(self, x):
        batch_feat, batch_label = self.data_compress(x)

        N = batch_label.shape[0]  # Batch size
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=self.opt.device)
        noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # alpha_bar_t       [N, 1, 1, 1]
        noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
        noise = torch.randn_like(batch_label)  # epsilon           [N, 2, T, F]
        noisy_audio = noise_scale_sqrt * (batch_label) + (1.0 - noise_scale) ** 0.5 * noise

        if self.opt.c_gen:
            # with torch.no_grad():
            condition = self.c_gen(batch_feat)['est_comp']
        else:
            condition = batch_feat

        out = self.model_ddpm(noisy_audio, condition, t)
        loss = com_mse_loss(out['est_noise'], noise, x['frame_num_list'])
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
                self.model_ddpm.train()
                loss_list = []
                for batch in self.train_loader:
                    # cuda
                    for key in batch.keys():
                        try:
                            batch[key] = batch[key].to(self.opt.device)
                        except AttributeError:
                            continue
                    out = self.run_step(batch)
                    self.optim_ddpm.zero_grad()
                    out['loss'].backward()
                    self.optim_ddpm.step()
                    self.progress.advance(batch_train, advance=1)
                    if self.opt.wandb:
                        wandb.log(
                            {
                                'train_loss': out['loss'].item()
                            }
                        )
                    else:
                        loss_list.append(out['loss'].item())
                    # break
                self.logger.info({
                    f'epoch_{epoch}: train_loss': np.mean(loss_list),
                })
                mean_valid_loss = self.inference()

                '''Adjust the learning rate and early stop'''
                if self.opt.half_lr > 1:
                    if mean_valid_loss >= prev_cv_loss:
                        cv_no_impv += 1
                        if cv_no_impv == self.opt.half_lr:
                            harving = True
                        if cv_no_impv >= self.opt.early_stop > 0:
                            self.console.print("No improvement and apply early stop")
                            return
                    else:
                        cv_no_impv = 0

                if harving == True:
                    optim_state = self.optim_ddpm.state_dict()
                    for i in range(len(optim_state['param_groups'])):
                        optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                    self.optim_ddpm.load_state_dict(optim_state)
                    self.console.print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                    harving = False
                prev_cv_loss = mean_valid_loss

                if mean_valid_loss < best_cv_loss:
                    self.console.print(
                        f"best loss is: {best_cv_loss}, current loss is: {mean_valid_loss}, save best_checkpoint.pth")
                    best_cv_loss = mean_valid_loss

                    self.save_cpt(epoch,
                                  save_path=f'./asset/model/'
                                            f'{self.model_ddpm.__class__.__name__}_best.pth')
                self.save_cpt(epoch,
                              save_path=f'./asset/model/'
                                        f'{self.model_ddpm.__class__.__name__}_{epoch}.pth')

    @torch.no_grad()
    def inference(self):
        self.model_ddpm.eval()
        if self.progress:
            loss = self.inference_()
        else:
            with Progress() as self.progress:
                loss = self.inference_()
        return loss

    @torch.no_grad()
    def inference_(self):
        loss_list = []
        # csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = [], [], [], [], [], []
        batch_valid = self.progress.add_task(f"[green]validating...", total=len(self.valid_loader))
        for batch in self.valid_loader:
            # cuda
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.opt.device)
                except AttributeError:
                    continue
            out = self.run_step(batch)
            loss_list.append(out['loss'].item())
            self.progress.advance(batch_valid, advance=1)
            # break
        test_loss = np.mean(loss_list)
        if self.opt.wandb:
            wandb.log({
                    'test_loss': test_loss
            })
        else:
            self.logger.info({
                'test_loss': test_loss
            })

        return test_loss

    @torch.no_grad()
    def inference_ddpm(self):
        csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = [], [], [], [], [], []
        alpha, beta, alpha_cum, sigmas, T = self.inference_schedule(fast_sampling=self.params.fast_sampling)
        with Progress() as self.progress:
            batch_valid_ddpm = self.progress.add_task(f"[green]validating...", total=len(self.valid_loader))
            for batch in self.valid_loader:
                # cuda
                for key in batch.keys():
                    try:
                        batch[key] = batch[key].to(self.opt.device)
                    except AttributeError:
                        continue
                # discard run_step function
                batch_feat, batch_label = self.data_compress(batch)
                if self.opt.c_gen:
                    # with torch.no_grad():
                    condition = self.c_gen(batch_feat)['est_comp']
                else:
                    condition = batch_feat
                # condition = batch_feat
                spec = torch.randn_like(condition)
                temp = spec  # for draw
                N = batch_label.shape[0]  # Batch size
                for n in range(len(alpha) - 1, -1, -1):
                    t = torch.tensor([T[n]], device=spec.device).repeat(N)
                    out = self.model_ddpm(spec, condition, t)

                    c1 = 1 / alpha[n] ** 0.5  # for ddpm sampling
                    c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5  # for ddpm sampling
                    spec = c1 * (spec - c2 * out['est_noise'])
                    if n > 0:  # + random noise
                        noise = torch.randn_like(spec)
                        spec += sigmas[n] * noise
                    # spec = torch.clamp(spec, -1, 1)

                '''run code in below can draw the difference between initial gaussian, condition (noisy), generated, and ground truth'''

                # spec = self.data_reconstuct(spec)
                # condition = self.data_reconstuct(condition)
                f, axs = plt.subplots(2, 4, figsize=(16, 6))

                axs[0, 0].imshow(temp[0, 0, :, :].cpu().numpy())
                axs[0, 0].set_title('n_real.png')
                axs[0, 0].axis('off')
                axs[0, 1].imshow(temp[0, 1, :, :].cpu().numpy())
                axs[0, 1].set_title('n_imag.png')
                axs[0, 1].axis('off')
                axs[0, 2].imshow(condition[0, 0, :, :].cpu().numpy())
                axs[0, 2].set_title('c_real.png')
                axs[0, 2].axis('off')
                axs[0, 3].imshow(condition[0, 1, :, :].cpu().numpy())
                axs[0, 3].set_title('c_imag.png')
                axs[0, 3].axis('off')


                axs[1, 0].imshow(spec[0, 0, :, :].cpu().numpy())
                axs[1, 0].set_title('g_real.png')
                axs[1, 0].axis('off')
                axs[1, 1].imshow(spec[0, 1, :, :].cpu().numpy())
                axs[1, 1].set_title('g_imag.png')
                axs[1, 1].axis('off')
                axs[1, 2].imshow(batch_label[0, 0, :, :].cpu().numpy())
                axs[1, 2].set_title('l_real.png')
                axs[1, 2].axis('off')
                axs[1, 3].imshow(batch_label[0, 1, :, :].cpu().numpy())
                axs[1, 3].set_title('l_imag.png')
                axs[1, 3].axis('off')
                plt.savefig('asset/data/sample_diffusec.jpg', dpi=300, bbox_inches='tight')
                exit()

                '''run code in above can draw the difference between initial gaussian, condition (noisy), generated, and ground truth'''

                batch_result = compare_complex(spec, batch_label,
                                               batch['frame_num_list'],
                                               feat_type=self.opt.feat_type)
                csig_list.append(batch_result[0])
                cbak_list.append(batch_result[1])
                covl_list.append(batch_result[2])
                pesq_list.append(batch_result[3])
                ssnr_list.append(batch_result[4])
                stoi_list.append(batch_result[5])

                self.progress.advance(batch_valid_ddpm, advance=1)

            print({
                'test_mean_csig': np.mean(csig_list),
                'test_mean_cbak': np.mean(cbak_list),
                'test_mean_covl': np.mean(covl_list),
                'test_mean_pesq': np.mean(pesq_list),
                'test_mean_ssnr': np.mean(ssnr_list),
                'test_mean_stoi': np.mean(stoi_list),
            })

    def save_cpt(self, step, save_path):
        super().save_cpt(step, save_path)
