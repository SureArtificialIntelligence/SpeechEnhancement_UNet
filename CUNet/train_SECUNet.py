import torch
import torchaudio
from DataETL import DataETL, DataLoader
from SE_CUNet import SECUNet
from pars_complexUNet_light import pars_gh as pars
from complex_blocks.wSDRLoss import wSDRLoss
from torch.optim import Adam
from os.path import join as pjoin

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

parser.add_argument('--monitor_seen_wav', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/CUNet/wSDR/monitor_seen_wav')
parser.add_argument('--monitor_unseen_wav', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/CUNet/wSDR/monitor_unseen_wav')
parser.add_argument('--save_models', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/CUNet/wSDR/saved_models')

parser.add_argument('--feature', type=str, default='time-domain')
parser.add_argument('--slice_win', type=int, default=16384)
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--fs', type=int, default=16000)

parser.add_argument('--bs_train', type=int, default=10)
parser.add_argument('--bs_valid', type=int, default=1)
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


def create_filenames(root_dir, filenames, postfix):
    filenames = [pjoin(root_dir, filename) + postfix for filename in filenames]
    return filenames


def print_flags():
    print('--------------------------- Flags -----------------------------')
    for flag in vars(args):
        print('{} : {}'.format(flag, getattr(args, flag)))
    print('{} : {}'.format('device', device))
    print('Actual batch size {} x {} = {}'.format(args.bs_train, args.num_segments, args.bs_train * args.num_segments))


if __name__ == '__main__':
    print('---------------------------------- Data Preparation -----------------------------')
    data_tr = DataETL('train', signal_filelist_path=args.train_speech_seeds, noise_filelist_path=args.train_noise_seeds,
                      feature=args.feature, short_version=False, slice_win=args.slice_win, num_segments=1,
                      mute_random=False, mute_random_snr=False, padding_slice=False, visualise=False)
    dl_tr = DataLoader(data_tr, shuffle=True, batch_size=args.bs_train, num_workers=16, drop_last=False)

    data_va = DataETL('valid', signal_filelist_path=args.valid_speech_seeds, noise_filelist_path=args.valid_noise_seeds,
                      feature=args.feature, slice_win=args.slice_win,
                      mute_random=True, mute_random_snr=True, padding_slice=False, visualise=True)
    dl_va = DataLoader(data_va, shuffle=False, batch_size=args.bs_valid, num_workers=0, drop_last=False)

    print('---------------------------------- Build Neural Networks ------------------------')
    se_cunet = SECUNet(pars)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    se_cunet = to_device(se_cunet, device)
    print(se_cunet)

    if args.optimiser == 'amsgrad':
        optimiser_G = Adam(se_cunet.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    else:
        optimiser_G = Adam(se_cunet.parameters(), lr=args.lr, betas=(0.5, 0.999))

    print_flags()

    print('------------------------------------- Start Training ----------------------------')
    loss_tr = 0.
    counter_tr = 0
    global_steps = 0
    for epoch in range(args.Epoch):
        for idx, batch_tr in enumerate(dl_tr):
            batch_noisy_mag = stack_batches(batch_tr['features_segments']['mixed'], split='train', num_segments=args.num_segments)
            batch_clean_mag = stack_batches(batch_tr['features_segments']['signal'], split='train', num_segments=args.num_segments)
            batch_clean_mag, batch_noisy_mag = to_device([batch_clean_mag, batch_noisy_mag], device)

            optimiser_G.zero_grad()
            denoised = se_cunet(batch_noisy_mag)

            loss = wSDRLoss(batch_noisy_mag.squeeze(), batch_clean_mag.squeeze(), denoised)
            loss.backward()
            optimiser_G.step()

            loss_tr += loss.item()
            counter_tr += 1
            global_steps += 1

            if global_steps % args.print_every == 0:
                avg_loss_tr = loss_tr/counter_tr
                print('[{} {}] {}'.format(epoch, idx, avg_loss_tr))

                if avg_loss_tr < args.loss_threshold:
                    for i in range(args.num_monitor):
                        clean_wav = batch_clean_mag[i, 0, 0, :]
                        noisy_wav = batch_noisy_mag[i, 0, 0, :]
                        denoised_wav = denoised[i, :]

                        clean_wav, noisy_wav, denoised_wav = from_device([clean_wav, noisy_wav, denoised_wav])

                        clean_path, noisy_path, denoised_path = create_filenames(args.monitor_seen_wav,
                                                                                 ['clean', 'noisy', 'denoised'],
                                                                                  '_{}_{}.wav'.format(global_steps, i))

                        [torchaudio.save(p, wav, args.fs)
                         for p, wav in
                         zip([clean_path, noisy_path, denoised_path],
                             [clean_wav, noisy_wav, denoised_wav])]

                counter_tr = 0
                loss_tr = 0.

            if global_steps % args.save_and_test_every == 0 and avg_loss_tr < args.loss_threshold:
                print('=> Saving model - {}'.format(global_steps))
                torch.save(se_cunet, pjoin(args.save_models, 'seunet_{}.pkl'.format(global_steps)))
                print('-------------------------------- Validation ---------------------------')
                loss_va = 0.
                counter_va = 0
                for idx_va, (batch_va, batch_info) in enumerate(dl_va):
                    mixed_segments_ = stack_batches(batch_va['features_segments']['mixed'], split='valid')
                    signal_segments_ = stack_batches(batch_va['features_segments']['signal'], split='valid')
                    noise_segments_ = stack_batches(batch_va['features_segments']['noise'], split='valid')
                    target_segments_ = stack_batches(batch_va['features_segments']['target'], split='valid')

                    mixed_segments_, signal_segments_ = to_device([mixed_segments_, signal_segments_], device)

                    with torch.no_grad():
                        denoised_ = se_cunet(mixed_segments_)

                    mixed_segments_ = mixed_segments_.squeeze()
                    signal_segments_ = signal_segments_.squeeze()
                    num_segments_va = denoised_.size()[0]
                    if num_segments_va == 1:
                        mixed_segments_ = mixed_segments_.unsqueeze(0)
                        signal_segments_ = signal_segments_.unsqueeze(0)

                    loss_v = wSDRLoss(mixed_segments_, signal_segments_, denoised_)
                    # print('Validation loss: {}'.format(loss_v.item()))
                    loss_va += loss_v
                    counter_va += 1
                    for i in range(num_segments_va):
                        signal_wav_va = signal_segments_[i, :]
                        mixed_wav_va = mixed_segments_[i, :]
                        denoised_wav_va = denoised_[i, :]
                        # print(signal_wav_va.requires_grad)
                        # print(mixed_wav_va.requires_grad)
                        # print(denoised_wav_va.requires_grad)
                        signal_wav_va, mixed_wav_va, denoised_wav_va = from_device([signal_wav_va, mixed_wav_va, denoised_wav_va])
                        # print(signal_wav_va.requires_grad)
                        # print(mixed_wav_va.requires_grad)
                        # print(denoised_wav_va.requires_grad)
                        clean_path_va, noisy_path_va, denoised_path_va = create_filenames(args.monitor_unseen_wav,
                                                                                          ['clean', 'noisy', 'denoised'],
                                                                                          '_{}_{}_{}.wav'.format(global_steps, idx_va, i))

                        [torchaudio.save(p, wav, args.fs)
                         for p, wav in
                         zip([clean_path_va, noisy_path_va, denoised_path_va],
                             [signal_wav_va, mixed_wav_va, denoised_wav_va])]

                print('==> Average validation loss: {}'.format(loss_va/counter_va))
                print()
