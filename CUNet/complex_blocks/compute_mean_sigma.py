import numpy as np
import torch
import os
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clean_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_trainset_56spk_wav_16k')
parser.add_argument('--noisy_data16_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_trainset_56spk_wav_16k')
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_spectrum_dir_5frames')
parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--win_len', type=int, default=400, help='windows length')    # 16k x 25ms
parser.add_argument('--hop_len', type=int, default=160, help='hop size')          # 16k x 10ms
args = parser.parse_args()


def normalize_wave_minmax(x):
    return (2. / 65536.) * (x - 32768.) + 1.


def slice_signal(path, win_len, hop_len, sampling_rate):
    # amp2db = AmplitudeToDB(stype='magnitude')
    sr, wavform = wavread(path)
    assert sampling_rate == sr
    wavform = torch.from_numpy(normalize_wave_minmax(wavform))
    stft_complex = torch.stft(wavform, win_len, hop_len)
    stft_real, stft_imag = stft_complex[:, :, 0], stft_complex[:, :, 1]
    return stft_real, stft_imag


def get_global_mu_sigma(clean_dir, noisy_dir, win_len, hop_len, sampling_rate):
    clean_real, clean_imag, noisy_real, noisy_imag = [], [], [], []
    i = 0
    for root, dirs, files in os.walk(clean_dir):
        for file in files:

            if file.endswith('.wav'):
                print('Processing {}'.format(file))
                clean_path, noisy_path = [pjoin(wav_dir, file) for wav_dir in [clean_dir, noisy_dir]]
                clean_slices, noisy_slices = [slice_signal(path, win_len, hop_len, sampling_rate)
                                              for path in [clean_path, noisy_path]]
                if len(clean_slices[0]) > 0 and len(noisy_slices[0]) > 0:
                    i = i + 1
                    for idx, (clean_slice, noisy_slice) in enumerate(zip(clean_slices, noisy_slices)):
                        clean_slice_real, clean_slice_imag = clean_slice[0].numpy().flatten().tolist(), clean_slice[1].numpy().flatten().tolist()
                        noisy_slice_real, noisy_slice_imag = noisy_slice[0].numpy().flatten().tolist(), noisy_slice[1].numpy().flatten().tolist()

                        clean_real.extend(clean_slice_real)
                        clean_imag.extend(clean_slice_imag)
                        noisy_real.extend(noisy_slice_real)
                        noisy_imag.extend(noisy_slice_imag)

            if i > 1000:
                break

    mu_clean_real, sigma_clean_real, max_clean_real, min_clean_real = np.mean(clean_real), np.std(clean_real), np.max(clean_real), np.min(clean_real)
    mu_clean_imag, sigma_clean_imag, max_clean_imag, min_clean_imag = np.mean(clean_imag), np.std(clean_imag), np.max(clean_imag), np.min(clean_imag)
    mu_noisy_real, sigma_noisy_real, max_noisy_real, min_noisy_real = np.mean(noisy_real), np.std(noisy_real), np.max(noisy_real), np.min(noisy_real)
    mu_noisy_imag, sigma_noisy_imag, max_noisy_imag, min_noisy_imag = np.mean(noisy_imag), np.std(noisy_imag), np.max(noisy_imag), np.min(noisy_imag)

    print('Input scaling')
    print('Real part -- mu:{} sigma:{} max:{} min:{}'.format(mu_noisy_real, sigma_noisy_real, max_noisy_real, min_noisy_real))
    print('Imag part -- mu:{} sigma:{} max:{} min:{}'.format(mu_noisy_imag, sigma_noisy_imag, max_noisy_imag, min_noisy_imag))

    print('Output scaling')
    print('Real part -- mu:{} sigma:{} max:{} min:{}'.format(mu_clean_real, sigma_clean_real, max_clean_real, min_clean_real))
    print('Imag part -- mu:{} sigma:{} max:{} min:{}'.format(mu_clean_imag, sigma_clean_imag, max_clean_imag, min_clean_imag))

    [max_noisy_real, min_noisy_real, max_noisy_imag, min_noisy_imag, max_clean_real, min_clean_real, max_clean_imag, min_clean_imag] = \
        [np.ceil(extremevalue) if extremevalue > 0 else np.floor(extremevalue) for extremevalue in
         [max_noisy_real, min_noisy_real, max_noisy_imag, min_noisy_imag, max_clean_real, min_clean_real, max_clean_imag, min_clean_imag]]

    clean_real_norm = (np.array(clean_real) - min_clean_real - (max_clean_real - min_clean_real) / 2) / ((max_clean_real - min_clean_real) / 2)
    clean_imag_norm = (np.array(clean_imag) - min_clean_imag - (max_clean_imag - min_clean_imag) / 2) / ((max_clean_imag - min_clean_imag) / 2)
    noisy_real_norm = (np.array(noisy_real) - min_noisy_real - (max_noisy_real - min_noisy_real) / 2) / ((max_noisy_real - min_noisy_real) / 2)
    noisy_imag_norm = (np.array(noisy_imag) - min_noisy_imag - (max_noisy_imag - min_noisy_imag) / 2) / ((max_noisy_imag - min_noisy_imag) / 2)

    print('----------------------------------------------------------------------------')
    print('clean real: {} {}'.format(max_clean_real, min_clean_real))
    print('clean imag: {} {}'.format(max_clean_imag, min_clean_imag))
    print('noisy real: {} {}'.format(max_noisy_real, min_noisy_real))
    print('noisy imag: {} {}'.format(max_noisy_imag, min_noisy_imag))
    print('----------------------------------------------------------------------------')

    # num_bins = 200
    # xrange = [-25, 25]
    # xnormrange = [-1, 1]
    # plt.figure(1)
    # plt.subplot(121)
    # plt.hist(clean_real, bins=num_bins, range=xrange)
    # plt.subplot(122)
    # plt.hist(clean_real_norm, bins=num_bins, range=xnormrange)
    # plt.figure(2)
    # plt.subplot(121)
    # plt.hist(clean_imag, bins=num_bins, range=xrange)
    # plt.subplot(122)
    # plt.hist(clean_imag_norm, bins=num_bins, range=xnormrange)
    # plt.figure(3)
    # plt.subplot(121)
    # plt.hist(noisy_real, bins=num_bins, range=xrange)
    # plt.subplot(122)
    # plt.hist(noisy_real_norm, bins=num_bins, range=xnormrange)
    # plt.figure(4)
    # plt.subplot(121)
    # plt.hist(noisy_imag, bins=num_bins, range=xrange)
    # plt.subplot(122)
    # plt.hist(noisy_imag_norm, bins=num_bins, range=xnormrange)
    # plt.show()

    clean_real = clean_real_norm
    clean_imag = clean_imag_norm
    noisy_real = noisy_real_norm
    noisy_imag = noisy_imag_norm
    mu_clean_real, sigma_clean_real, max_clean_real, min_clean_real = np.mean(clean_real), np.std(clean_real), np.max(clean_real), np.min(clean_real)
    mu_clean_imag, sigma_clean_imag, max_clean_imag, min_clean_imag = np.mean(clean_imag), np.std(clean_imag), np.max(clean_imag), np.min(clean_imag)
    mu_noisy_real, sigma_noisy_real, max_noisy_real, min_noisy_real = np.mean(noisy_real), np.std(noisy_real), np.max(noisy_real), np.min(noisy_real)
    mu_noisy_imag, sigma_noisy_imag, max_noisy_imag, min_noisy_imag = np.mean(noisy_imag), np.std(noisy_imag), np.max(noisy_imag), np.min(noisy_imag)

    print('Input scaling')
    print('Real part -- mu:{} sigma:{} max:{} min:{}'.format(mu_noisy_real, sigma_noisy_real, max_noisy_real, min_noisy_real))
    print('Imag part -- mu:{} sigma:{} max:{} min:{}'.format(mu_noisy_imag, sigma_noisy_imag, max_noisy_imag, min_noisy_imag))

    print('Output scaling')
    print('Real part -- mu:{} sigma:{} max:{} min:{}'.format(mu_clean_real, sigma_clean_real, max_clean_real, min_clean_real))
    print('Imag part -- mu:{} sigma:{} max:{} min:{}'.format(mu_clean_imag, sigma_clean_imag, max_clean_imag, min_clean_imag))

    return clean_real, clean_imag, noisy_real, noisy_imag


if __name__ == '__main__':
    clean_real, clean_imag, noisy_real, noisy_imag = get_global_mu_sigma(
        args.clean_data16_dir, args.noisy_data16_dir, args.win_len, args.hop_len, args.fs)

