import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.io.wavfile import write as wavwrite

segment_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_MagPhase_observation'
files = os.listdir(segment_dir)
files = [pjoin(segment_dir, f) for f in files]

files_name = [f.split('/')[-1].split('.')[0] for f in files]
files_name = list(set(files_name))

for file_name in files_name:
    id_files = [f for f in files if f.split('/')[-1].split('.')[0].startswith(file_name)]
    print('Processing {}'.format(file_name))
    centers_clean_mag, centers_clean_pha, centers_noisy_mag, centers_noisy_pha, centers_noise_mag, centers_noise_pha = [], [], [], [], [], []
    for idx in range(len(id_files)):
        file = pjoin(segment_dir, '{}.wav_{}.npy'.format(file_name, idx))
        segment = np.load(file)
        segment_clean_mag = segment[0]
        segment_clean_pha = segment[1]
        center_clean_mag = segment_clean_mag[:, 2]
        center_clean_pha = segment_clean_pha[:, 2]
        centers_clean_mag.append(center_clean_mag)
        centers_clean_pha.append(center_clean_pha)

        segment_noisy_mag = segment[2]
        segment_noisy_pha = segment[3]
        center_noisy_mag = segment_noisy_mag[:, 2]
        center_noisy_pha = segment_noisy_pha[:, 2]
        centers_noisy_mag.append(center_noisy_mag)
        centers_noisy_pha.append(center_noisy_pha)

    clean_mag = np.stack(centers_clean_mag).T
    clean_pha = np.stack(centers_clean_pha).T

    noisy_mag = np.stack(centers_noisy_mag).T
    noisy_pha = np.stack(centers_noisy_pha).T

    clean_mag = np.exp(clean_mag)
    clean_stft = clean_mag * np.exp(1j * clean_pha)
    clean_real, clean_imag = torch.real(torch.from_numpy(clean_stft)), torch.imag(torch.from_numpy(clean_stft))
    clean_stft = torch.stack([clean_real, clean_imag], dim=-1)
    # clean = np.stack([clean_mag, np.expand_dims(clean_pha, axis=0)], -1)
    clean_wav = torch.istft(clean_stft, 400, 160)
    wavwrite('../save_wav/test_clean.wav', 16000, clean_wav.numpy())

    noisy_mag = np.exp(noisy_mag)
    noisy_stft = noisy_mag * np.exp(1j * noisy_pha)
    noisy_real, noisy_imag = torch.real(torch.from_numpy(noisy_stft)), torch.imag(torch.from_numpy(noisy_stft))
    noisy_stft = torch.stack([noisy_real, noisy_imag], dim=-1)
    noisy_wav = torch.istft(noisy_stft, 400, 160)
    wavwrite('../save_wav/test_noisy.wav', 16000, noisy_wav.numpy())

    # timestamps = len(id_files)
    #
    # plt.figure()
    # plt.subplot(221)
    # plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(clean_mag)))
    # plt.subplot(222)
    # plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(clean_pha)))
    # plt.subplot(223)
    # plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(noisy_mag)))
    # plt.subplot(224)
    # plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(noisy_pha)))
    # plt.show()


