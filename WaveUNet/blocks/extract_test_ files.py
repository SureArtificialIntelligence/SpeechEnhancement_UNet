import os
from os.path import join as pjoin

test_dir = '/nas/staff/data_work/Sure/GPU_wav_dump/SEGAN/monitor_test'
tgt_dir = '/home/user/Desktop/wocao2'
prefix_noisy, prefix_noise, prefix_clean, prefix_denoised = 'noisy_', 'noise_', 'clean_', 'denoised_'

file_id = 'p257_427.wav_1'

prefix_noisy, prefix_noise, prefix_clean = [pf + file_id + '.wav' for pf in [prefix_noisy, prefix_noise, prefix_clean]]

noisy_wav = pjoin(test_dir, prefix_noisy)
noise_wav = pjoin(test_dir, prefix_noise)
clean_wav = pjoin(test_dir, prefix_clean)
denoised_wav = pjoin(test_dir, prefix_denoised+file_id+'_35000.wav')

tgt_noisy_wav = pjoin(tgt_dir, prefix_noisy)
tgt_noise_wav = pjoin(tgt_dir, prefix_noise)
tgt_clean_wav = pjoin(tgt_dir, prefix_clean)
tgt_denoised_wav = pjoin(tgt_dir, prefix_denoised+file_id+'_35000.wav')


os.system('cp {} {}'.format(noisy_wav, tgt_noisy_wav))
os.system('cp {} {}'.format(noise_wav, tgt_noise_wav))
os.system('cp {} {}'.format(clean_wav, tgt_clean_wav))
os.system('cp {} {}'.format(denoised_wav, tgt_denoised_wav))

