import math
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

torchaudio.set_audio_backend("soundfile")  # default backend (SoX) has bugs when loading WAVs


class WAVDataset(Dataset):
    """
    Create a PyTorch Dataset object from a directory containing clean and noisy WAV files
    """
    def __init__(self, dir: Path, n_fft, test=False):
        self.clean_dir = dir.joinpath('clean')
        self.noisy_dir = dir.joinpath('noisy')
        self.n_fft = n_fft
        self.test = test

        assert os.path.exists(self.clean_dir), 'No clean WAV file folder found!'
        assert os.path.exists(self.noisy_dir), 'No noisy WAV file folder found!'

        self.clean_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.clean_dir))):
            self.clean_WAVs[i] = self.clean_dir.joinpath(filename)

        self.noisy_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.noisy_dir))):
            self.noisy_WAVs[i] = self.noisy_dir.joinpath(filename)

        # VAD
        step = 16000 / self.n_fft
        frequency_bins = np.arange(0, (self.n_fft // 2 + 1) * step, step=step)
        self.VAD_frequencies = np.where((frequency_bins >= 300) & (frequency_bins <= 5000), True, False)

        # mean normalization
        frameshift = (self.n_fft / 16000) / 4  # frameshift between STFT frames
        t_init = 0.1  # init time
        tauFeat = 3  # time constant
        tauFeatInit = 0.1  # time constant during init
        self.n_init_frames = math.ceil(t_init / frameshift)
        self.alpha_feat_init = math.exp(-frameshift / tauFeatInit)
        self.alpha_feat = math.exp(-frameshift / tauFeat)

    def __len__(self):
        return len(self.noisy_WAVs)

    def __getitem__(self, idx):
        noisy_path = self.noisy_WAVs[idx]
        clean_path = self.clean_dir.joinpath(noisy_path.name.split('+')[0] + '.wav')  # get the filename of the clean WAV from the filename of the noisy WAV
        while True:
            try:
                clean_waveform, _ = torchaudio.load(clean_path, normalization=2**15)
                noisy_waveform, _ = torchaudio.load(noisy_path, normalization=2**15)
            except (RuntimeError, OSError):
                continue
            break

        assert clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 1, 'WAV file is not single channel!'

        window = torch.hamming_window(self.n_fft)
        x_stft = torch.stft(noisy_waveform.view(-1), n_fft=self.n_fft, hop_length=self.n_fft // 4, win_length=self.n_fft, window=window)
        y_stft = torch.stft(clean_waveform.view(-1), n_fft=self.n_fft, hop_length=self.n_fft // 4, win_length=self.n_fft, window=window)

        x_ps = x_stft.pow(2).sum(-1)
        x_lps = LogTransform()(x_ps)

        x_ms = x_ps.sqrt()
        y_ms = y_stft.pow(2).sum(-1).sqrt()

        # VAD
        y_ms_filtered = y_ms[self.VAD_frequencies]
        y_energy_filtered = y_ms_filtered.pow(2).mean(dim=0)
        y_energy_filtered_averaged = self.__moving_average(y_energy_filtered)
        y_peak_energy = y_energy_filtered_averaged.max()
        VAD = torch.where(y_energy_filtered_averaged > y_peak_energy / 1000, torch.ones_like(y_energy_filtered), torch.zeros_like(y_energy_filtered))
        VAD = VAD.bool()

        # mean normalization
        frames = []
        x_lps = x_lps.transpose(0, 1)  # (time, frequency)
        n_init_frames = self.n_init_frames
        alpha_feat_init = self.alpha_feat_init
        alpha_feat = self.alpha_feat
        for frame_counter, frame_feature in enumerate(x_lps):
            if frame_counter < n_init_frames:
                alpha = alpha_feat_init
            else:
                alpha = alpha_feat
            if frame_counter == 0:
                mu = frame_feature
                sigmasquare = frame_feature.pow(2)
            mu = alpha * mu + (1 - alpha) * frame_feature
            sigmasquare = alpha * sigmasquare + (1 - alpha) * frame_feature.pow(2)
            sigma = torch.sqrt(torch.clamp(sigmasquare - mu.pow(2), min=1e-12))  # limit for sqrt
            norm_feature = (frame_feature - mu) / sigma
            frames.append(norm_feature)

        x_lps = torch.stack(frames, dim=0).transpose(0, 1)   # (frequency, time)

        if not self.test:
            return x_lps, x_ms, y_ms, VAD
        if self.test:
            return noisy_waveform.view(-1), clean_waveform.view(-1), x_stft, y_stft, x_lps, x_ms, y_ms, VAD
    
    def __moving_average(self, a, n=3):
        ret = torch.cumsum(a, dim=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n - 1] = a[:n - 1]
        ret[n - 1:] = ret[n - 1:] / n
        return ret


class LogTransform(torch.nn.Module):
    def __init__(self, floor=10**-12):
        super().__init__()
        self.floor = floor

    def forward(self, specgram):
        return torch.log(torch.clamp(specgram, min=self.floor))
