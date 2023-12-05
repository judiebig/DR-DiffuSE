from torch.utils.data import Dataset
import glob
import os
import torch.nn as nn
import random
import torch
import numpy as np
import librosa


class ToTensor(object):
    def __call__(self, x, tensor_type='float'):
        if tensor_type == 'float':
            return torch.FloatTensor(x)
        elif tensor_type == 'int':
            return torch.IntTensor(x)


class BatchInfo(object):
    def __init__(self, noisy, clean, frame_num_list, wav_len_list):
        self.feats = noisy
        self.labels = clean
        self.frame_num_list = frame_num_list
        self.wav_len_list = wav_len_list


class CustomCollate(object):
    def __init__(self, opt):
        self.win_size = opt.win_size
        self.fft_num = opt.fft_num
        self.win_shift = opt.win_shift

    @staticmethod
    def normalize(x):
        return x / np.max(abs(x))

    # return {
    #     'noisy_speech': noisy,
    #     'clean_speech': clean,
    #     'frame_num': frame_num,
    #     'wav_len': wav_len
    # }

    def collate_fn(self, batch):
        noisy_list, clean_list, frame_num_list, wav_len_list = [], [], [], []
        to_tensor = ToTensor()
        for sample in batch:
            c = np.sqrt(len(sample['noisy_speech']) / np.sum(sample['noisy_speech'] ** 2.0))
            noisy_list.append(to_tensor(sample['noisy_speech'] * c))
            clean_list.append(to_tensor(sample['clean_speech'] * c))
            frame_num_list.append(sample['frame_num'])
            wav_len_list.append(sample['wav_len'])
        noisy_list = nn.utils.rnn.pad_sequence(noisy_list, batch_first=True)
        clean_list = nn.utils.rnn.pad_sequence(clean_list, batch_first=True)  # [b, chunk_length]
        noisy_list = torch.stft(
            noisy_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num),
            return_complex=False
        ).permute(0, 3, 2, 1)  # [b, 2, T, F] real tensor, return_complex = false
        clean_list = torch.stft(
            clean_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num),
            return_complex=False
        ).permute(0, 3, 2, 1)  # [b, 2, T, F]

        return {
            'feats': noisy_list,
            'labels': clean_list,
            'frame_num_list': frame_num_list,
            'wav_len_list': wav_len_list
        }


class VBDataset(Dataset):
    def __init__(self, noisy_root, clean_root, data_type, opt):
        super(VBDataset, self).__init__()
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.chunk_length = opt.chunk_length
        self.win_size = opt.win_size
        self.fft_num = opt.fft_num
        self.win_shift = opt.win_shift
        self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]

        assert data_type in ['train', 'valid']
        self.data_type = data_type  # determine train or test

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        noisy, _ = librosa.load(os.path.join(self.noisy_root, self.raw_paths[index]), sr=16000)
        clean, _ = librosa.load(os.path.join(self.clean_root, self.raw_paths[index]), sr=16000)
        if self.data_type == 'train':
            if len(noisy) > self.chunk_length:
                wav_start = random.randint(0, len(noisy) - self.chunk_length)
                noisy = noisy[wav_start:wav_start + self.chunk_length]
                clean = clean[wav_start:wav_start + self.chunk_length]
        wav_len = len(noisy)
        frame_num = (len(noisy) - self.win_size + self.fft_num) // self.win_shift + 1
        # return noisy, clean, frame_num, wav_len
        return {
            'noisy_speech': noisy,
            'clean_speech': clean,
            'frame_num': frame_num,
            'wav_len': wav_len
        }