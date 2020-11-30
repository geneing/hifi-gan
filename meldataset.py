import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.fileio.sound_scp import SoundScpReader

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


# def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#     if torch.min(y) < -1.:
#         print('min value is ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is ', torch.max(y))

#     global mel_basis, hann_window
#     if fmax not in mel_basis:
#         mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
#         mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
#         hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

#     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
#     y = y.squeeze(1)

#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True)

#     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

#     spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
#     spec = spectral_normalize_torch(spec)

#     return spec


# def get_dataset_filelist(a):
#     with open(a.input_training_file, 'r', encoding='utf-8') as fi:
#         training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
#                           for x in fi.read().split('\n') if len(x) > 0]

#     with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
#         validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
#                             for x in fi.read().split('\n') if len(x) > 0]
#     return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, scp_file, feature_stats_file_path, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False):
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = 0 if fmin is None else fmin
        self.fmax = sampling_rate / 2 if fmax is None else fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning

        self.root_path = root_path
        self.scp_file_path = root_path+scp_file
        self.loader = SoundScpReader(self.scp_file_path, root_path=self.root_path, normalize=True, always_2d=False, sampling_rate=self.sampling_rate)
        self.logmel_layer = LogMelFbank(fs=sampling_rate,
                                   n_fft=n_fft,
                                   win_length=win_size,
                                   hop_length=hop_size,
                                   center=True,
                                   normalized=False,
                                   onesided=True,
                                   n_mels=num_mels,
                                   fmin=fmin,
                                   fmax=fmax,
                                   htk=False)
        self.logmel_layer_loss = LogMelFbank(fs=sampling_rate,
                                        n_fft=n_fft,
                                        win_length=win_size,
                                        hop_length=hop_size,
                                        center=True,
                                        normalized=False,
                                        onesided=True,
                                        n_mels=num_mels,
                                        fmin=fmin,
                                        fmax=fmax_loss,
                                        htk=False)
        self.norm_layer = GlobalMVN(stats_file=feature_stats_file_path,
                                    norm_means=True,
                                    norm_vars=True)

        self.keys = list(self.loader.keys())

    # def keys(self):
    #     return list(self.loader.keys())

    def __len__(self):
        return len(self.keys)

    # def __iter__(self):
    #     return iter(self.loader)

    def logmelfilterbank(self, audio,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     eps=1e-10, mel_basis=None
                     ):
        """Compute log-Mel filterbank feature.

        Args:
            audio (ndarray): Audio signal (T,).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length. If set to None, it will be the same as fft_size.
            window (str): Window function type.
            eps (float): Epsilon value to avoid inf in log calculation.

        Returns:
            ndarray: Log Mel filterbank feature (#frames, num_mels).

        """
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                            win_length=win_length, window=window, pad_mode="reflect")
        spc = np.abs(x_stft).T  # (#frames, #bins)

        return np.log10(np.maximum(eps, np.dot(spc, mel_basis)))


    def mel_spectrograms(self, audio):
        # extract feature
        mel, _ = self.logmel_layer(torch.FloatTensor(audio).unsqueeze(0))
        mel_loss, _ = self.logmel_layer_loss(torch.FloatTensor(audio).unsqueeze(0))

        # make sure the audio length and feature length are matched
        n_mel = mel.shape[1] #number of mel frames
        audio = np.pad(audio, (0, self.n_fft), mode="reflect")
        audio = audio[:n_mel * self.hop_size]
        assert n_mel * self.hop_size == len(audio)

        normed_mel, _       = self.norm_layer(torch.FloatTensor(mel))
        normed_mel_loss, _  = self.norm_layer(torch.FloatTensor(mel_loss))

        return torch.FloatTensor(audio), torch.transpose(normed_mel, 1, 2), torch.transpose(normed_mel_loss, 1, 2)


    def mel_spectrogram_loss(self, audio):
        # extract feature
        # print(audio.shape)

        mel_loss, _ = self.logmel_layer_loss(audio)

        # make sure the audio length and feature length are matched
        # audio = torch.functional.pad(audio, (0, self.n_fft), mode="reflect")
        # audio = audio[:len(mel_loss) * self.hop_size]
        # assert len(mel_loss) * self.hop_size == len(audio)

        normed_mel_loss, _ = self.norm_layer(mel_loss)

        return torch.transpose(normed_mel_loss, 1, 2)

    def __getitem__(self, key:str):

        if self._cache_ref_count == 0:
            sampling_rate, audio = self.loader[self.keys[key]]
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # make sure audio is a multiple of segment size to avoid mel size mismatch
        pad_size = audio.shape[0] % self.win_size
        if pad_size>0:
            audio = np.pad(audio, (0, pad_size), 'constant')

        #audio = audio.unsqueeze(0)
        if not self.fine_tuning:
            if self.split:
                if audio.shape[0] >= self.segment_size:
                    max_audio_start = audio.shape[0] - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(0)), 'constant')

            audio, mel, mel_loss = self.mel_spectrograms(audio)

            #print("getitem:245\t", audio.shape, mel.shape)

        else:
            assert False
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.FloatTensor(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            mel_loss = mel

        return (mel.squeeze(), audio,
                self.loader.get_path(self.keys[key]), mel_loss.squeeze())
