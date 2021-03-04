import time
import math
import numpy as np
import torch
import os
from torchaudio.functional import magphase
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread, write as wavwrite


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clean_data16_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_trainset_56spk_wav_16k')
parser.add_argument('--noisy_data16_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_trainset_56spk_wav_16k')
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_MagPhase')

parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--win_len', type=int, default=400, help='windows length')    # 16k x 25ms
parser.add_argument('--hop_len', type=int, default=160, help='hop size')          # 16k x 10ms
parser.add_argument('--win_frames', type=int, default=5, help='windows frames')  # 10 * 40 + 15 = 415ms
parser.add_argument('--hop_frames', type=int, default=1, help='hop frames')
args = parser.parse_args()


def normalize_wave_minmax(x):
    return (2. / 65536.) * (x - 32768.) + 1.


def slice_signal(path, win_len, hop_len, win_frames, hop_frames, sampling_rate):
    slices = []
    sr, wavform = wavread(path)
    assert sampling_rate == sr
    wavform = torch.from_numpy(normalize_wave_minmax(wavform))
    stft_complex = torch.stft(wavform, win_len, hop_len)
    # stft_mag_orig, stft_pha_orig = stft_complex[:, :, 0].numpy(), stft_complex[:, :, 1].numpy()
    mag, pha = magphase(stft_complex)
    mag = torch.log(mag + 1e-7)
    # stft_mag = in_mag_scale(mag)
    # stft_pha = in_pha_scale(stft_pha_orig)

    # print(np.max(np.abs(stft_mag_recover - stft_mag_orig)))
    # assert stft_mag_recover.all() == stft_mag_orig.all()
    # assert stft_pha_recover.all() == stft_pha_orig.all()
    # stft_mag_recover = inverse_in_mag_scale(stft_mag)
    # stft_pha_recover = inverse_in_pha_scale(stft_pha)
    #
    # stft_recover = np.stack([stft_mag_recover, stft_pha_recover], axis=-1)
    # signal_recover = torch.istft(torch.from_numpy(stft_recover), n_fft=400, hop_length=160)
    # wavwrite('./recover.wav', 16000, signal_recover.numpy())
    # stft_orig = np.stack([stft_mag_orig, stft_pha_orig], axis=-1)
    # signal_orig = torch.istft(torch.from_numpy(stft_orig), n_fft=400, hop_length=160)
    # wavwrite('./orig.wav', 16000, signal_orig.numpy())

    len_frames = stft_complex.size()[-2]
    num_slices = math.floor((len_frames - win_frames) / hop_frames) + 1
    if num_slices > 0:
        for idx_slice in range(num_slices):
            slices.append([mag[:, idx_slice * hop_frames:idx_slice * hop_frames + win_frames],
                           pha[:, idx_slice * hop_frames:idx_slice * hop_frames + win_frames]])
            # slices_pha.append(stft_pha[:, idx_slice * hop_frames : idx_slice * hop_frames + win_frames].numpy())
    return slices


def process(clean_dir, noisy_dir, save_to, win_len, hop_len, win_frames, hop_frames, sampling_rate, num_utterances):
    counter = 0
    for root, dirs, files in os.walk(clean_dir[:num_utterances]):
        for file in files:
            if file.endswith('.wav'):
                counter += 1
                print('Processing {}({}/{})'.format(file, counter, num_utterances))
                # ss = time.time()
                clean_path, noisy_path = [pjoin(wav_dir, file) for wav_dir in [clean_dir, noisy_dir]]
                clean_slices, noisy_slices = [slice_signal(path, win_len, hop_len, win_frames, hop_frames, sampling_rate)
                                              for path in [clean_path, noisy_path]]
                if len(clean_slices[0]) > 0 and len(noisy_slices[0]) > 0:
                    for idx, (clean_slice, noisy_slice) in enumerate(zip(clean_slices, noisy_slices)):
                        clean_slice_mag, clean_slice_pha = clean_slice[0], clean_slice[1]
                        noisy_slice_mag, noisy_slice_pha = noisy_slice[0], noisy_slice[1]

                        # noisy_slice_mag_db = amp2db(noisy_slice_mag)
                        # print(noisy_slice_mag_db)
                        # [clean_slice_mag, clean_slice_pha,
                        #  noisy_slice_mag, noisy_slice_pha,
                        #  noise_slice_mag, noise_slice_pha] = [np.log(cpn + 1e-5) for cpn in
                        #                                         [clean_slice_mag, clean_slice_pha,
                        #                                          noisy_slice_mag, noisy_slice_pha,
                        #                                          noise_slice_mag, noise_slice_pha]]
                        group = np.stack([clean_slice_mag, clean_slice_pha,
                                          noisy_slice_mag, noisy_slice_pha], axis=0)
                        np.save(pjoin(save_to, '{}_{}.npy'.format(file, idx)), group)
                        # to = time.time()
                        # duration = to - ss
                        # print('Using time: {}'.format(duration))
                else:
                    print('Not sufficient length!')


def check_processed_files(data_dir):
    filenames = os.listdir(data_dir)
    unsatisfied = []
    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        file_content = np.load(filepath)
        if file_content.shape != (6, 201, 40):
            unsatisfied.append(filename)
            print('{} has shape of {}'.format(filename, file_content.shape))
    print('Unsatisfied files {}'.format(len(unsatisfied)))
    print(unsatisfied)


if __name__ == '__main__':
    process(args.clean_data16_dir, args.noisy_data16_dir, args.processed_data_dir,
            win_len=args.win_len, hop_len=args.hop_len, win_frames=args.win_frames, hop_frames=args.hop_frames,
            sampling_rate=args.fs, num_utterances=1000)
    # check_processed_files(args.processed_data_dir)
