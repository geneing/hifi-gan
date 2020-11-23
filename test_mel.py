#%%
import os
import numpy as np
import torch

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.fileio.sound_scp import SoundScpReader

#%%
root_path = "/rhome/eingerman/Neural/MachineCodes/Speech/ESPnet/egs2/blizzard2013/tts1/"
# "/rhome/eingerman/Neural/MachineCodes/Speech/ESPnet/egs2/blizzard2013/tts1/"
path = root_path+"/dump/raw/tr_no_dev/wav.scp"
#"/data/train/wav.scp"

os.chdir(root_path)
loader = SoundScpReader(path, root_path=root_path, normalize=True, always_2d=False)

for k in loader.keys():
    fs, sounds = loader[k]
    break

#%%
x = torch.from_numpy(sounds[np.newaxis, :]).to(torch.float32)
#   (feats_extract): LogMelFbank(
#     (stft): Stft(n_fft=1024, win_length=1024, hop_length=256, center=True, normalized=False, onesided=True)
#     (logmel): LogMel(sr=22050, n_fft=1024, n_mels=80, fmin=80, fmax=7600, htk=False)
#   )
# (normalize): GlobalMVN(stats_file=exp/tts_train_raw_phn_tacotron_g2p_en/decode_use_teacher_forcingtrue_train.loss.ave/stats/train/feats_stats.npz, norm_means=True, norm_vars=True)

logmel_layer = LogMelFbank(fs=fs, n_fft=1024, win_length=1024, hop_length=256, center=True,
                    normalized=False, onesided=True, n_mels=80, fmin=80, fmax=7600, htk=False)
y, _ = logmel_layer(x)

#%%
norm_layer = GlobalMVN(stats_file=root_path+"exp/tts_train_raw_phn_tacotron_g2p_en/decode_use_teacher_forcingtrue_train.loss.ave/stats/train/feats_stats.npz", norm_means=True, norm_vars=True)
z, _ = norm_layer(y)
# %%
