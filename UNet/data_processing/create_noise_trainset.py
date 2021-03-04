import os
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread, write as wavwrite
clean_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/clean_testset_wav_16k'
noisy_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/noisy_testset_wav_16k'
noise_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/noise_testset_wav_16k'

filenames = os.listdir(clean_dir)
num_filenames = len(filenames)
file_counter = 0

for filename in filenames:
    file_counter += 1
    print('Processing audio file [{}/{}]: {}'.format(file_counter, num_filenames, filename))
    clean_path = pjoin(clean_dir, filename)
    noisy_path = pjoin(noisy_dir, filename)
    noise_path = pjoin(noise_dir, filename)
    fs, clean_waveform = wavread(clean_path)
    _, noisy_waveform = wavread(noisy_path)
    noise_waveform = noisy_waveform - clean_waveform
    wavwrite(noise_path, fs, noise_waveform)
