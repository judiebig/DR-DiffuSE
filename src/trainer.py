from abc import abstractmethod

import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from rich.progress import Progress

import warnings

warnings.filterwarnings('ignore')


class BasicTrainer:
    def __init__(self, train_data, valid_data, model, logger, opt):
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
        self.model = model.to(opt.device)

        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # others
        self.opt = opt
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
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()
            },
            save_path
        )


class VBTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, logger, opt):
        super(VBTrainer, self).__init__(train_data, valid_data, model, logger, opt)

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
        batch_feat, batch_label = self.data_compress(x)
        out = self.model(batch_feat)
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
                self.model.train()
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
                            self.logger.print("No improvement and apply early stop")
                            return
                    else:
                        cv_no_impv = 0

                if harving == True:
                    optim_state = self.optim.state_dict()
                    for i in range(len(optim_state['param_groups'])):
                        optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                    self.optim.load_state_dict(optim_state)
                    self.logger.print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                    harving = False
                prev_cv_loss = mean_valid_loss

                if mean_valid_loss < best_cv_loss:
                    self.logger.print(
                        f"best loss is: {best_cv_loss}, current loss is: {mean_valid_loss}, save best_checkpoint.pth")
                    best_cv_loss = mean_valid_loss

                    self.save_cpt(epoch,
                                  save_path=f'./asset/model/'
                                            f'{self.model.__class__.__name__}_best.pth')
                self.save_cpt(epoch,
                              save_path=f'./asset/model/'
                                        f'{self.model.__class__.__name__}_{epoch}.pth')

    @torch.no_grad()
    def inference(self):
        self.model.eval()
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
