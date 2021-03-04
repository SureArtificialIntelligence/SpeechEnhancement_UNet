import time
import math
import numpy as np
import torch
import os
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread, write as wavwrite
from data_processing.scaling_spectrum import in_real_scale, in_imag_scale, out_real_scale, out_imag_scale, \
    inverse_in_real_scale, inverse_in_imag_scale, inverse_out_real_scale, inverse_out_imag_scale

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clean_data16_dir', type=str,
                        default='/nas/staff/data_work/Sure/Edinburg_Speech/clean_trainset_56spk_wav_16k')
parser.add_argument('--noisy_data16_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/noisy_trainset_56spk_wav_16k')
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_RealImag_shortversion')  # processed_spectrum_dir_5frames
parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--win_len', type=int, default=400, help='windows length')    # 16k x 25ms
parser.add_argument('--hop_len', type=int, default=160, help='hop size')          # 16k x 10ms
parser.add_argument('--win_frames', type=int, default=5, help='windows frames')  # 10 * 40 + 15 = 415ms
parser.add_argument('--hop_frames', type=int, default=1, help='hop frames')
args = parser.parse_args()


def normalize_wave_minmax(x):
    return (2. / 65536.) * (x - 32768.) + 1.


def slice_signal(path, win_len, hop_len, win_frames, hop_frames, sampling_rate, stream):
    slices = []
    sr, wavform = wavread(path)
    assert sampling_rate == sr
    wavform = torch.from_numpy(normalize_wave_minmax(wavform))
    stft_complex = torch.stft(wavform, win_len, hop_len)
    stft_real_orig, stft_imag_orig = stft_complex[:, :, 0].numpy(), stft_complex[:, :, 1].numpy()

    assert stream in ['in', 'out']
    if stream == 'in':
        stft_real = in_real_scale(stft_real_orig)
        stft_imag = in_imag_scale(stft_imag_orig)
    else:
        stft_real = out_real_scale(stft_real_orig)
        stft_imag = out_imag_scale(stft_imag_orig)

    # print(np.max(np.abs(stft_real_recover - stft_real_orig)))
    # assert stft_real_recover.all() == stft_real_orig.all()
    # assert stft_imag_recover.all() == stft_imag_orig.all()
    # stft_real_recover = inverse_in_real_scale(stft_real)
    # stft_imag_recover = inverse_in_imag_scale(stft_imag)
    #
    # stft_recover = np.stack([stft_real_recover, stft_imag_recover], axis=-1)
    # signal_recover = torch.istft(torch.from_numpy(stft_recover), n_fft=400, hop_length=160)
    # wavwrite('./recover.wav', 16000, signal_recover.numpy())
    # stft_orig = np.stack([stft_real_orig, stft_imag_orig], axis=-1)
    # signal_orig = torch.istft(torch.from_numpy(stft_orig), n_fft=400, hop_length=160)
    # wavwrite('./orig.wav', 16000, signal_orig.numpy())



    # stft_real = inverse_out_real_scale(stft_real)
    # stft_imag = inverse_out_imag_scale(stft_imag)
    # stft = np.stack([np.expand_dims(stft_real, axis=0), np.expand_dims(stft_imag, axis=0)], axis=-1)

    len_frames = stft_complex.size()[-2]
    num_slices = math.floor((len_frames - win_frames) / hop_frames) + 1
    if num_slices > 0:
        for idx_slice in range(num_slices):
            slices.append([stft_real[:, idx_slice * hop_frames:idx_slice * hop_frames + win_frames],
                           stft_imag[:, idx_slice * hop_frames:idx_slice * hop_frames + win_frames]])
            # slices_imag.append(stft_imag[:, idx_slice * hop_frames : idx_slice * hop_frames + win_frames].numpy())
    # num_slices = len(slices)
    # slices_real, slices_imag = [], []
    # for idx in range(num_slices):
    #     slice_real = slices[idx][0][:, 2]
    #     slice_imag = slices[idx][1][:, 2]
    #     slices_real.append(slice_real)
    #     slices_imag.append(slice_imag)
    #
    # stft_real = np.stack(slices_real)
    # stft_imag = np.stack(slices_imag)
    #
    # stft_real = inverse_out_real_scale(stft_real).T
    # stft_imag = inverse_out_imag_scale(stft_imag).T
    # stft = np.stack([np.expand_dims(stft_real, axis=0), np.expand_dims(stft_imag, axis=0)], axis=-1)
    # wav = torch.istft(torch.from_numpy(stft), 400, 160)
    # wavwrite('../save_wav/test2.wav', 16000, wav.numpy().T)
    return slices


def process(clean_dir, noisy_dir, save_to, win_len, hop_len, win_frames, hop_frames, sampling_rate):
    for root, dirs, files in os.walk(clean_dir):
        for file in files:
            if file.endswith('.wav'):
                print('Processing {}'.format(file))
                ss = time.time()
                clean_path, noisy_path = [pjoin(wav_dir, file) for wav_dir in [clean_dir, noisy_dir]]
                clean_slices, noisy_slices = [slice_signal(path, win_len, hop_len, win_frames, hop_frames, sampling_rate, stream)
                                              for path, stream in zip([clean_path, noisy_path], ['out', 'in'])]

                num_slices = len(noisy_slices)
                slices_real, slices_imag = [], []
                for idx in range(num_slices):
                    slice_real = noisy_slices[idx][0][:, 2]
                    slice_imag = noisy_slices[idx][1][:, 2]
                    slices_real.append(slice_real)
                    slices_imag.append(slice_imag)

                stft_real = np.stack(slices_real)
                stft_imag = np.stack(slices_imag)

                stft_real = inverse_in_real_scale(stft_real).T
                stft_imag = inverse_in_imag_scale(stft_imag).T
                stft = np.stack([np.expand_dims(stft_real, axis=0), np.expand_dims(stft_imag, axis=0)], axis=-1)
                wav = torch.istft(torch.from_numpy(stft), 400, 160)
                wavwrite('../save_wav/test3.wav', 16000, wav.numpy().T)

                num_slices = len(clean_slices)
                slices_real, slices_imag = [], []
                for idx in range(num_slices):
                    slice_real = clean_slices[idx][0][:, 2]
                    slice_imag = clean_slices[idx][1][:, 2]
                    slices_real.append(slice_real)
                    slices_imag.append(slice_imag)

                stft_real = np.stack(slices_real)
                stft_imag = np.stack(slices_imag)

                stft_real = inverse_out_real_scale(stft_real).T
                stft_imag = inverse_out_imag_scale(stft_imag).T
                stft = np.stack([np.expand_dims(stft_real, axis=0), np.expand_dims(stft_imag, axis=0)], axis=-1)
                wav = torch.istft(torch.from_numpy(stft), 400, 160)
                wavwrite('../save_wav/test4.wav', 16000, wav.numpy().T)

                if len(clean_slices[0]) > 0 and len(noisy_slices[0]) > 0:
                    for idx, (clean_slice, noisy_slice) in enumerate(zip(clean_slices, noisy_slices)):
                        clean_slice_real, clean_slice_imag = clean_slice[0], clean_slice[1]
                        noisy_slice_real, noisy_slice_imag = noisy_slice[0], noisy_slice[1]
                        noise_slice_real, noise_slice_imag = \
                            noisy_slice_real - clean_slice_real, noisy_slice_imag - clean_slice_imag
                        # noisy_slice_real_db = amp2db(noisy_slice_real)
                        # print(noisy_slice_real_db)
                        # [clean_slice_real, clean_slice_imag,
                        #  noisy_slice_real, noisy_slice_imag,
                        #  noise_slice_real, noise_slice_imag] = [np.log(cpn + 1e-5) for cpn in
                        #                                         [clean_slice_real, clean_slice_imag,
                        #                                          noisy_slice_real, noisy_slice_imag,
                        #                                          noise_slice_real, noise_slice_imag]]
                        group = np.stack([clean_slice_real, clean_slice_imag,
                                          noisy_slice_real, noisy_slice_imag,
                                          noise_slice_real, noise_slice_imag], axis=0)
                        np.save(pjoin(save_to, '{}_{}.npy'.format(file, idx)), group)
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
        if file_content.shape != (6, 201, 5):
            unsatisfied.append(filename)
            print('{} has shape of {}'.format(filename, file_content.shape))
    print('Unsatisfied files {}'.format(len(unsatisfied)))
    print(unsatisfied)


if __name__ == '__main__':
    process(args.clean_data16_dir, args.noisy_data16_dir, args.processed_data_dir,
            win_len=args.win_len, hop_len=args.hop_len, win_frames=args.win_frames, hop_frames=args.hop_frames,
            sampling_rate=args.fs)
    check_processed_files(args.processed_data_dir)
