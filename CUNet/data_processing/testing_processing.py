import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
from data_processing.scaling_spectrum import inverse_in_real_scale, inverse_in_imag_scale, inverse_out_real_scale, inverse_out_imag_scale
from scipy.io.wavfile import write as wavwrite

segment_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_RealImag_observation/example_long'
files = os.listdir(segment_dir)
files = [pjoin(segment_dir, f) for f in files]

files_name = [f.split('/')[-1].split('.')[0] for f in files]
files_name = list(set(files_name))

for file_name in files_name:
    id_files = [f for f in files if f.split('/')[-1].split('.')[0].startswith(file_name)]
    print('Processing {}'.format(file_name))
    centers_clean_real, centers_clean_imag, centers_noisy_real, centers_noisy_imag, centers_noise_real, centers_noise_imag = [], [], [], [], [], []
    for idx in range(len(id_files)):
        file = pjoin(segment_dir, '{}.wav_{}.npy'.format(file_name, idx))
        segment = np.load(file)
        segment_clean_real = segment[0]
        segment_clean_imag = segment[1]
        center_clean_real = segment_clean_real[:, 2]
        center_clean_imag = segment_clean_imag[:, 2]
        centers_clean_real.append(center_clean_real)
        centers_clean_imag.append(center_clean_imag)

        segment_noisy_real = segment[2]
        segment_noisy_imag = segment[3]
        center_noisy_real = segment_noisy_real[:, 2]
        center_noisy_imag = segment_noisy_imag[:, 2]
        centers_noisy_real.append(center_noisy_real)
        centers_noisy_imag.append(center_noisy_imag)

    clean_real = np.stack(centers_clean_real)
    clean_imag = np.stack(centers_clean_imag)
    clean_real = inverse_out_real_scale(clean_real).T
    clean_imag = inverse_out_imag_scale(clean_imag).T

    noisy_real = np.stack(centers_noisy_real)
    noisy_imag = np.stack(centers_noisy_imag)
    noisy_real = inverse_in_real_scale(noisy_real).T
    noisy_imag = inverse_in_imag_scale(noisy_imag).T

    clean = np.stack([np.expand_dims(clean_real, axis=0), np.expand_dims(clean_imag, axis=0)], -1)
    clean_wav = torch.istft(torch.from_numpy(clean), 400, 160)
    wavwrite('../save_wav/test_clean.wav', 16000, clean_wav.numpy().T)

    noisy = np.stack([np.expand_dims(noisy_real, axis=0), np.expand_dims(noisy_imag, axis=0)], -1)
    noisy_wav = torch.istft(torch.from_numpy(noisy), 400, 160)
    wavwrite('../save_wav/test_noisy.wav', 16000, noisy_wav.numpy().T)

    timestamps = len(id_files)

    plt.figure()
    plt.subplot(221)
    plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(clean_real)))
    plt.subplot(222)
    plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(clean_imag)))
    plt.subplot(223)
    plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(noisy_real)))
    plt.subplot(224)
    plt.pcolormesh(range(timestamps), range(201), np.log(np.abs(noisy_imag)))
    plt.show()


