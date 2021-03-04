"""
Downsampling 48kHz -> 16kHz
Slicing
"""
import os
import math
import numpy as np
import time
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread

import argparse

Training = False
Test = True
parser = argparse.ArgumentParser()
if Training:
    parser.add_argument('--clean_data48_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_trainset_56spk_wav')
    parser.add_argument('--noisy_data48_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_trainset_56spk_wav')
    parser.add_argument('--clean_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_trainset_56spk_wav_16k')
    parser.add_argument('--noisy_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_trainset_56spk_wav_16k')

    parser.add_argument('--processed_data_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir')

if Test:
    parser.add_argument('--clean_data48_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_testset_wav')
    parser.add_argument('--noisy_data48_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_testset_wav')
    parser.add_argument('--clean_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_testset_wav_16k')
    parser.add_argument('--noisy_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_testset_wav_16k')

    parser.add_argument('--processed_data_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_test')


parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--win_len', type=int, default=16384, help='sampling rate')
parser.add_argument('--hop_len', type=int, default=8192, help='sampling rate')
args = parser.parse_args()


def downsample(src_dir, dst_dir, sampling_rate):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.wav'):
                src_path = pjoin(root, file)
                dst_path = pjoin(dst_dir, file)
                print('Processing {}'.format(src_path))
                sox_command = 'sox {} -r {} {}'.format(src_path, sampling_rate, dst_path)
                os.system(sox_command)


def slice_signal(wav_path, win_len, hop_len, sampling_rate):
    slices = []
    sr, wavform = wavread(wav_path)
    assert sr == sampling_rate
    len_wav = len(wavform)
    num_slices = math.floor((len_wav - win_len)/hop_len) + 1
    if num_slices > 0:
        for idx_slice in range(num_slices):
            slice = wavform[idx_slice * hop_len:(idx_slice * hop_len + win_len)]
            slices.append(slice)
    return slices


def process(clean_dir, noisy_dir, save_to, win_len, hop_len, sampling_rate):
    for root, dirs, files in os.walk(clean_dir):
        for file in files:
            if file.endswith('.wav'):
                print('Processing {}'.format(file))
                ss = time.time()
                clean_path, noisy_path = [pjoin(wav_dir, file) for wav_dir in [clean_dir, noisy_dir]]
                clean_slices, noisy_slices = [slice_signal(path, win_len, hop_len, sampling_rate)
                                              for path in [clean_path, noisy_path]]
                if len(clean_slices) > 0 and len(noisy_slices) > 0:
                    for idx, (clean_slice, noisy_slice) in enumerate(zip(clean_slices, noisy_slices)):
                        pair = np.array([clean_slice, noisy_slice])
                        np.save(pjoin(save_to, '{}_{}.npy'.format(file, idx)), pair)
                        to = time.time()
                        duration = to - ss
                        print('Using time: {}'.format(duration))
                else:
                    print('Not sufficient length!')


def check_processed_files(data_dir):
    filenames = os.listdir(data_dir)
    unsatisfied = []
    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        file_content = np.load(filepath)
        if file_content.shape != (2, 16384):
            unsatisfied.append(filename)
            print('{} has shape of {}'.format(filename, file_content.shape))
    print('Unsatisfied files {}'.format(len(unsatisfied)))
    print(unsatisfied)


if __name__ == '__main__':
    downsample(args.clean_data48_dir, args.clean_data16_dir, args.fs)
    downsample(args.noisy_data48_dir, args.noisy_data16_dir, args.fs)
    process(args.clean_data16_dir, args.noisy_data16_dir, args.processed_data_dir, args.win_len, args.hop_len, args.fs)
    check_processed_files(args.processed_data_dir)
