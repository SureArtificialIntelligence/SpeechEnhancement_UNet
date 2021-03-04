import torch
import torchaudio
from DataETL import DataETL, DataLoader
from visualiseData import DataVisualisation
from SE_UNet import SEUNet
from pars_UNet_light import pars_gh as pars
from blocks.wSDRLoss import wSDRLoss
from torch.optim import Adam
import os
from os.path import join as pjoin
from show_denoising_results import show_results

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--train_speech_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/train.pkl')
parser.add_argument('--train_noise_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/train.pkl')
parser.add_argument('--valid_speech_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/valid.pkl')
parser.add_argument('--valid_noise_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/valid.pkl')
parser.add_argument('--test_speech_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/test.pkl')
parser.add_argument('--test_noise_seeds', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/test.pkl')

parser.add_argument('--monitor_wav', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SEUNet/wi_wsdr/monitor_wav')
parser.add_argument('--monitor_wav_te', type=str,
                    default='/home/user/Desktop/huawei_demo')
parser.add_argument('--save_models', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/UNet/wi_wsdr/saved_models')

parser.add_argument('--feature', type=str, default='time-domain')
parser.add_argument('--slice_win', type=int, default=16384)
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--fs', type=int, default=16000)

parser.add_argument('--bs_train', type=int, default=10)
parser.add_argument('--bs_valid', type=int, default=1)
parser.add_argument('--bs_test', type=int, default=1)
parser.add_argument('--optimiser', type=str, default='amsgrad')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--Epoch', type=int, default=10000)

parser.add_argument('--num_monitor', type=int, default=5)
parser.add_argument('--print_every', type=int, default=200)
parser.add_argument('--save_and_test_every', type=int, default=2000)
parser.add_argument('--loss_threshold', type=float, default=-0.97)
args = parser.parse_args()


def to_device(elements, device):
    if isinstance(elements, list):
        elements = [element.to(device) for element in elements]
    else:
        elements = elements.to(device)
    return elements


def from_device(elements):
    if isinstance(elements, list):
        elements = [element.detach().cpu() for element in elements]
    else:
        elements = elements.detach().cpu()
    return elements


def create_filenames(root_dir, filenames, postfix):
    filenames = [pjoin(root_dir, filename) + postfix for filename in filenames]
    return filenames


def stack_batches(tensor_list, split='train', num_segments=None):
    """

    :param tensor_list: tensor list contains N tensor, each tensor contains (bs, feature maps)
    :param split: train/valid/test
    :param num_segments: N
    :return: (N x bs, feature maps)
    """

    if split == 'train':
        assert num_segments is not None
        if len(tensor_list[0].shape) == 3:
            tensor_list = [tensor.unsqueeze(1) for tensor in tensor_list]
        bs_each = [tensor_list[seg_idx].shape[0] for seg_idx in range(num_segments)]

        bs_cutoff = 0
        bs_seg = [0]
        for bs in bs_each:
            bs_cutoff += bs
            bs_seg.append(bs_cutoff)

        new_bs = sum(bs_each)
        new_shape = [new_bs]
        [new_shape.append(shape_element) for shape_element in tensor_list[0].shape[1:]]
        merge_tensor = torch.zeros(new_shape)
        for idx, tensor in enumerate(tensor_list):
            merge_tensor[bs_seg[idx]:bs_seg[idx + 1]] = tensor

    else:
        merge_tensor = torch.stack([mx for mx in tensor_list], dim=0)

    return merge_tensor


def combine_wavs(wav_dir):
    wavs = [wav for wav in os.listdir(wav_dir) if wav.startswith('denoised_')]
    max_idx = max([int(wav.split('_')[-1][:-4]) for wav in wavs if wav.startswith('denoised_')])
    denoised_id = wavs[0][:-5]

    denoised_paths, noisy_paths, clean_paths = [], [], []
    command_denoised = 'sox'
    command_clean = 'sox'
    command_noisy = 'sox'
    command_noisy_denoised = 'sox'
    for idx in range(max_idx):
        denoised_path = pjoin(wav_dir, denoised_id + str(idx) + '.wav')
        noisy_path = pjoin(wav_dir, denoised_id.replace('denoised', 'noisy') + str(idx) + '.wav')
        clean_path = pjoin(wav_dir, denoised_id.replace('denoised', 'clean') + str(idx) + '.wav')

        denoised_paths.append(denoised_path)
        noisy_paths.append(noisy_path)
        clean_paths.append(clean_path)

        command_denoised = command_denoised + ' {}'.format(denoised_path)
        command_noisy = command_noisy + ' {}'.format(noisy_path)
        command_clean = command_clean + ' {}'.format(clean_path)

        if max_idx > 3:
            if idx < 2:
                command_noisy_denoised = command_noisy_denoised + ' {}'.format(noisy_path)
            else:
                command_noisy_denoised = command_noisy_denoised + ' {}'.format(denoised_path)

        else:
            if idx < 1:
                command_noisy_denoised = command_noisy_denoised + ' {}'.format(noisy_path)
            else:
                command_noisy_denoised = command_noisy_denoised + ' {}'.format(denoised_path)

    tgt_denoised_path = pjoin(wav_dir, 'denoised.wav')
    tgt_noisy_path = pjoin(wav_dir, 'noisy.wav')
    tgt_clean_path = pjoin(wav_dir, 'clean.wav')
    tgt_noisy_denoised_path = pjoin(wav_dir, 'noisy_denoised.wav')

    command_denoised = command_denoised + ' ' + tgt_denoised_path
    command_noisy = command_noisy + ' ' + tgt_noisy_path
    command_clean = command_clean + ' ' + tgt_clean_path
    command_noisy_denoised = command_noisy_denoised + ' ' + tgt_noisy_denoised_path

    os.system(command_denoised)
    os.system(command_noisy)
    os.system(command_clean)
    os.system(command_noisy_denoised)


# data visualisation
data_te = DataETL('test', signal_filelist_path=args.test_speech_seeds, noise_filelist_path=args.test_noise_seeds,
                  feature=args.feature, short_version=2, slice_win=args.slice_win,
                  mute_random=True, mute_random_snr=True, padding_slice=False, visualise=True)

audio_save2 = './audio_demo/'
dv = DataVisualisation(data_te, audio_save2)
dv.visualise(num_segments=1)
dl_te = DataLoader(data_te, shuffle=False, batch_size=args.bs_test, num_workers=0, drop_last=False)

# model selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = 'seunet'
# model_type = 'secunet'
# model_type = 'wav_unet'
global_steps = 1100000
print('=> Loading model - {}'.format(global_steps))
ss = torch.load(pjoin(args.save_models, '{}_{}.pkl'.format(model_type, global_steps)), map_location=device)


ss = to_device(ss, device)

print('-------------------------------- Test ---------------------------')
loss_te = 0.
counter_te = 0

for idx_te, (batch_te, batch_info) in enumerate(dl_te):
    mixed_segments_ = stack_batches(batch_te['features_segments']['mixed'], split='test')
    signal_segments_ = stack_batches(batch_te['features_segments']['signal'], split='test')
    noise_segments_ = stack_batches(batch_te['features_segments']['noise'], split='test')
    target_segments_ = stack_batches(batch_te['features_segments']['target'], split='test')

    # mixed_segments_, signal_segments_ = to_device([mixed_segments_, signal_segments_], device)

    with torch.no_grad():
        denoised_ = ss(mixed_segments_)

    mixed_segments_ = mixed_segments_.squeeze()
    signal_segments_ = signal_segments_.squeeze()
    num_segments_va = denoised_.size()[0]
    if num_segments_va == 1:
        mixed_segments_ = mixed_segments_.unsqueeze(0)
        signal_segments_ = signal_segments_.unsqueeze(0)

    loss_v = wSDRLoss(mixed_segments_, signal_segments_, denoised_)
    # print('Validation loss: {}'.format(loss_v.item()))
    loss_te += loss_v
    counter_te += 1
    for i in range(num_segments_va):
        signal_wav_va = signal_segments_[i, :]
        mixed_wav_va = mixed_segments_[i, :]
        denoised_wav_va = denoised_[i, :]
        # signal_wav_va, mixed_wav_va, denoised_wav_va = from_device([signal_wav_va, mixed_wav_va, denoised_wav_va])

        clean_path_va, noisy_path_va, denoised_path_va = create_filenames(args.monitor_wav_te,
                                                                          ['clean', 'noisy', 'denoised'],
                                                                          '_{}_{}_{}.wav'.format(global_steps, idx_te,
                                                                                                 i))

        [torchaudio.save(p, wav, args.fs)
         for p, wav in
         zip([clean_path_va, noisy_path_va, denoised_path_va],
             [signal_wav_va, mixed_wav_va, denoised_wav_va])]

    combine_wavs(args.monitor_wav_te)

    show_results(args.monitor_wav_te)

print('==> Average validation loss: {}'.format(loss_te / counter_te))
