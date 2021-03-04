import socket
import torch
import random
import numpy as np
from os.path import join as pjoin
from AudioSlices_ETL import AudioSlicesETL, DataLoader
from SEGAN import Generator, Discriminator
from nn_pars import pars_generator, pars_discriminator
from blocks.EmphasisFilter import preemphais_filter, inv_preemphais_filter
from torch.optim import RMSprop, Adam
from scipy.io.wavfile import write as wavwrite
from torch.nn.utils import clip_grad_norm_

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--processed_data_dir_te', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_test')
parser.add_argument('--monitor_wav_te', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SEGAN/monitor_test/')
# parser.add_argument('--monitor_wav_te', type=str,
#                    default='/home/user/Desktop/wocao/')
parser.add_argument('--save_models', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SEGAN/saved_models_more_general/')
parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--bs', type=int, default=32, help='batch size')  # 32
parser.add_argument('--test_bs', type=int, default=256, help='batch size')  # 32
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# random seed
setup_seed(20)


def split_into_streams(batch):
    batch_pairs = preemphais_filter(batch.numpy())
    # rec = np.rint(inv_preemphais_filter(torch.from_numpy(preemphais_filter(batch.numpy())).type(torch.FloatTensor))).astype(np.int16)
    # print(batch.numpy())
    # print(rec)
    batch_clean, batch_noisy = batch_pairs[:, 0], batch_pairs[:, 1]
    batch_pairs, batch_clean, batch_noisy = [torch.from_numpy(stream).type(torch.FloatTensor)
                                             for stream in
                                             [batch_pairs, batch_clean, batch_noisy]]
    batch_clean, batch_noisy = [stream.unsqueeze(1) for stream in [batch_clean, batch_noisy]]
    # reconstructed = inv_preemphais_filter(batch_pairs.numpy())
    return batch_pairs, batch_clean, batch_noisy


def to_device(elements, device):
    if isinstance(elements, list):
        elements = [element.to(device) for element in elements]
    else:
        elements = elements.to(device)
    return elements


if __name__ == '__main__':
    # get monitor test
    for i in range(10):
        data_te = AudioSlicesETL(audioslices_dir=args.processed_data_dir_te)
        print('Getting Monitor Batch Test')
        monitor_batch_clean_te, monitor_batch_noisy_te, filename_te = data_te.monitor_eval_training(args.test_bs)
        monitor_batch_clean_te = torch.from_numpy(preemphais_filter(monitor_batch_clean_te.numpy()))
        monitor_batch_noisy_te = torch.from_numpy(preemphais_filter(monitor_batch_noisy_te.numpy()))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print(monitor_batch_clean[0])
        monitor_batch_clean_te, monitor_batch_noisy_te = [stream.type(torch.FloatTensor)
                                                          for stream in
                                                          [monitor_batch_clean_te, monitor_batch_noisy_te]]
        monitor_batch_clean_te, monitor_batch_noisy_te = to_device([monitor_batch_clean_te, monitor_batch_noisy_te], device)

        # save noisy and clean reference
        noisy_te = inv_preemphais_filter(monitor_batch_noisy_te.cpu().numpy())
        clean_te = inv_preemphais_filter(monitor_batch_clean_te.cpu().numpy())
        noise_te = noisy_te - clean_te
        for i in range(args.test_bs):
            noisy_path_te = pjoin(args.monitor_wav_te, 'noisy_{}.wav'.format(filename_te[i].split('/')[-1][:-4]))
            clean_path_te = pjoin(args.monitor_wav_te, 'clean_{}.wav'.format(filename_te[i].split('/')[-1][:-4]))
            noise_path_te = pjoin(args.monitor_wav_te, 'noise_{}.wav'.format(filename_te[i].split('/')[-1][:-4]))
            wavwrite(noisy_path_te, args.fs, noisy_te[i].T)
            wavwrite(clean_path_te, args.fs, clean_te[i].T)
            wavwrite(noise_path_te, args.fs, noise_te[i].T)

        random_z = to_device(torch.rand(args.bs, 1024, 8), device)
        L2_criteria = torch.nn.MSELoss()

        global_steps = 35000
        # test samples monitor
        random_z_ = random_z
        netG = torch.load(pjoin(args.save_models, 'netG_{}.pkl'.format(global_steps)))  #, map_location=torch.device('cpu'))
        gen_te = netG(random_z_, monitor_batch_noisy_te).detach()
        # error_left = L2_criteria(gen_, monitor_batch_noisy - monitor_batch_clean)
        error_left_te = L2_criteria(gen_te, monitor_batch_clean_te)
        error_rm_te = L2_criteria(gen_te, monitor_batch_noisy_te)
        # denoised_ = (monitor_batch_noisy - gen_)
        denoised_te = gen_te
        noise_te = monitor_batch_noisy_te - denoised_te
        # gen_ = inv_preemphais_filter(gen_.detach().cpu().numpy())
        noise_te = inv_preemphais_filter(noise_te.detach().cpu().numpy())
        denoised_te = inv_preemphais_filter(denoised_te.detach().cpu().numpy())
        print('Test Error after {} steps: {}, {}'.format(global_steps, error_left_te.item(), error_rm_te.item()))
        print('----------------------------------------------------')

        for i in range(args.test_bs):
            noise_path = pjoin(args.monitor_wav_te, 'noise_{}_{}.wav'.format(filename_te[i].split('/')[-1][:-4], global_steps))
            denoised_path = pjoin(args.monitor_wav_te, 'denoised_{}_{}.wav'.format(filename_te[i].split('/')[-1][:-4], global_steps))
            wavwrite(noise_path, args.fs, noise_te[i].T)
            wavwrite(denoised_path, args.fs, denoised_te[i].T)
