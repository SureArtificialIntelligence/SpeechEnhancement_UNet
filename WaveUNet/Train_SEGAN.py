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
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir')
#parser.add_argument('--processed_data_dir_te', type=str,
#                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_observation')
parser.add_argument('--processed_data_dir_te', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_test_shortversion')
parser.add_argument('--data_pickle', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/segan_list.pkl')

if socket.gethostname() == 'UAU-86505':  # local machine
    parser.add_argument('--save_models', type=str, default='./saved_models')
    parser.add_argument('--monitor_wav', type=str, default='./monitor_wav')
    parser.add_argument('--monitor_wav_te', type=str, default='./monitor_wav_test')
    parser.add_argument('--bs', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--print_every', type=int, default=10, help='save & print results')
    parser.add_argument('--save_every', type=int, default=50, help='save & print results')
else:
    parser.add_argument('--monitor_seen_wav', type=str,
                        default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/WaveUNet/MSE/monitor_seen_wav/')
    parser.add_argument('--monitor_unseen_wav', type=str,
                        default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/WaveUNet/MSE/monitor_unseen_wav')
    parser.add_argument('--save_models', type=str,
                        default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/WaveUNet/MSE/saved_models')
    parser.add_argument('--bs', type=int, default=128, help='batch size')   # 32
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--print_every', type=int, default=100, help='save & print results')   # 100
    parser.add_argument('--save_every', type=int, default=500, help='save & print results')   # 500

parser.add_argument('--optimiser', type=str, default='Adam', help='optimiser')
parser.add_argument('--lr_D', type=float, default=3e-4, help='learning rate for discriminator')
parser.add_argument('--lr_G', type=float, default=3e-4, help='learning rate for generator')
parser.add_argument('--Epochs', type=int, default=10000, help='batch size')
parser.add_argument('--l1_factor', type=int, default=100, help='batch size')
parser.add_argument('--monitor_bs', type=int, default=50, help='monitor batch size')
parser.add_argument('--fs', type=int, default=16000, help='sampling rate')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
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
    data_tr = AudioSlicesETL(audioslices_dir=args.processed_data_dir)
    dl_tr = DataLoader(data_tr, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    netG = Generator(pars_generator)
    netD = Discriminator(pars_discriminator)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG, netD = to_device([netG, netD], device)

    if args.optimiser == 'Adam':
        optimiser_G = Adam(netG.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
        optimiser_D = Adam(netD.parameters(), lr=args.lr_D, betas=(0.5, 0.999))
    elif args.optimiser == 'RMSprop':
        optimiser_G = RMSprop(netG.parameters(), lr=args.lr_G)
        optimiser_D = RMSprop(netD.parameters(), lr=args.lr_D)
    else:
        optimiser_G = RMSprop(netG.parameters(), lr=args.lr_G)
        optimiser_D = RMSprop(netD.parameters(), lr=args.lr_D)

    print('--------------------------- Flags -----------------------------')
    for flag in vars(args):
        print('{} : {}'.format(flag, getattr(args, flag)))
    print('{} : {}'.format('device', device))

    print('------------------------- Preparation -------------------------')
    # get reference batch
    print('Getting Reference Batch for Virtual Batch Norm')
    ref_batch = data_tr.get_reference(args.bs)
    ref_batch, ref_clean, ref_noisy = split_into_streams(ref_batch)
    ref_batch, ref_clean, ref_noisy = to_device([ref_batch, ref_clean, ref_noisy], device)

    # get monitor_training
    print('Getting Monitor Batch Train')
    monitor_batch_clean, monitor_batch_noisy, _ = data_tr.monitor_eval_training(args.monitor_bs)
    monitor_batch_clean = torch.from_numpy(preemphais_filter(monitor_batch_clean.numpy()))
    monitor_batch_noisy = torch.from_numpy(preemphais_filter(monitor_batch_noisy.numpy()))

    # print(monitor_batch_clean[0])
    monitor_batch_clean, monitor_batch_noisy = [stream.type(torch.FloatTensor)
                                                for stream in
                                                [monitor_batch_clean, monitor_batch_noisy]]
    monitor_batch_clean, monitor_batch_noisy = to_device([monitor_batch_clean, monitor_batch_noisy], device)

    # save noisy and clean reference
    noisy = inv_preemphais_filter(monitor_batch_noisy.cpu().numpy())
    clean = inv_preemphais_filter(monitor_batch_clean.cpu().numpy())
    noise = noisy - clean
    for i in range(args.monitor_bs):
        noisy_path = pjoin(args.monitor_seen_wav, 'noisy_{}.wav'.format(i))
        clean_path = pjoin(args.monitor_seen_wav, 'clean_{}.wav'.format(i))
        noise_path = pjoin(args.monitor_seen_wav, 'noise_{}.wav'.format(i))
        wavwrite(noisy_path, args.fs, noisy[i].T)
        wavwrite(clean_path, args.fs, clean[i].T)
        wavwrite(noise_path, args.fs, noise[i].T)

    # get monitor test
    data_te = AudioSlicesETL(audioslices_dir=args.processed_data_dir_te)
    print('Getting Monitor Batch Test')
    monitor_batch_clean_te, monitor_batch_noisy_te, _ = data_te.monitor_eval_training(20)
    monitor_batch_clean_te = torch.from_numpy(preemphais_filter(monitor_batch_clean_te.numpy()))
    monitor_batch_noisy_te = torch.from_numpy(preemphais_filter(monitor_batch_noisy_te.numpy()))

    # print(monitor_batch_clean[0])
    monitor_batch_clean_te, monitor_batch_noisy_te = [stream.type(torch.FloatTensor)
                                                for stream in
                                                [monitor_batch_clean_te, monitor_batch_noisy_te]]
    monitor_batch_clean_te, monitor_batch_noisy_te = to_device([monitor_batch_clean_te, monitor_batch_noisy_te], device)

    # save noisy and clean reference
    noisy_te = inv_preemphais_filter(monitor_batch_noisy_te.cpu().numpy())
    clean_te = inv_preemphais_filter(monitor_batch_clean_te.cpu().numpy())
    noise_te = noisy_te - clean_te
    for i in range(20):
        noisy_path_te = pjoin(args.monitor_unseen_wav, 'noisy_{}.wav'.format(i))
        clean_path_te = pjoin(args.monitor_unseen_wav, 'clean_{}.wav'.format(i))
        noise_path_te = pjoin(args.monitor_unseen_wav, 'noise_{}.wav'.format(i))
        wavwrite(noisy_path_te, args.fs, noisy_te[i].T)
        wavwrite(clean_path_te, args.fs, clean_te[i].T)
        wavwrite(noise_path_te, args.fs, noise_te[i].T)

    # ------------------------------------------- debug in middle ---------------------------------
    # noisy = np.rint(inv_preemphais_filter(monitor_batch_noisy.cpu().numpy())).astype(np.int16)
    # clean = np.rint(inv_preemphais_filter(monitor_batch_clean.cpu().numpy())).astype(np.int16)
    # global_steps = 0
    # i = 0
    # noisy_path = pjoin(args.monitor_wav, 'noisy_{}_{}.wav'.format(global_steps, i))
    # clean_path = pjoin(args.monitor_wav, 'clean_{}_{}.wav'.format(global_steps, i))
    # wavwrite(noisy_path, args.fs, noisy[i].T)
    # wavwrite(clean_path, args.fs, clean[i].T)
    # # --------------------------------------------------------------------------------------------
    L2_criteria = torch.nn.MSELoss()

    random_z = to_device(torch.rand(args.bs, 1024, 8), device)

    gradients = []
    print('-------------------------- Training ----------------------------')
    global_steps = 0
    for epoch in range(args.Epochs):
        for idx, batch_tr in enumerate(dl_tr):
            batch_pairs, batch_clean, batch_noisy = split_into_streams(batch_tr)
            batch_noise = batch_noisy - batch_clean
            batch_pairs, batch_clean, batch_noisy, batch_noise = to_device([batch_pairs, batch_clean, batch_noisy, batch_noise], device)

            # # train D
            # optimiser_D.zero_grad()
            # critic_clean = netD(batch_clean, batch_noisy, ref_batch)
            # loss_clean = torch.mean((critic_clean - 1.0) ** 2)
            #
            # denoised = netG(random_z, batch_noisy).detach()  # very important when training D
            # print(denoised)
            # print(ref_batch)
            # print(batch_noisy)
            #
            # critic_denoised = netD(denoised, batch_noisy, ref_batch)
            # loss_denoised = torch.mean((critic_denoised - 0.0) ** 2)
            #
            # loss_D = (loss_clean + loss_denoised) / 2
            # loss_D.backward()
            # optimiser_D.step()

            # train G
            optimiser_G.zero_grad()
            denoised_retry = netG(random_z, batch_noisy)
            # estimate_noise = netG(random_z, batch_noisy)
            # print(denoised_retry)
            # critic_denoised_retry = netD(denoised_retry, batch_noisy, ref_batch)
            # loss_G = torch.mean((critic_denoised_retry - 1.0) ** 2) / 2
            # L1_penalty = torch.mean(torch.abs(denoised_retry - batch_clean))
            L2_penalty = L2_criteria(denoised_retry, batch_clean)
            # L2_penalty = L2_criteria(estimate_noise, batch_noise)
            # loss_G = loss_G + L1_penalty * args.l1_factor
            loss_G = L2_penalty * args.l1_factor
            loss_G.backward()
            # clip_grad_norm_(netG.parameters(), args.clip)
            optimiser_G.step()

            # [gradients.extend(param.grad.view(-1).detach().cpu().numpy()) for param in netG.parameters() if param.grad is not None]
            # print(np.max(gradients))
            global_steps += 1
            if global_steps % args.print_every == 0:
                print('Loss_G: {}'.format(loss_G.item()))
            #     print('({})[{} {}] loss_D: {}, loss_G: {}, L1 Norm: {}, Critic - Clean: {}, Denoised: {}/{}'.format(
            #         global_steps, epoch, idx, loss_D.item(), loss_G.item(), L1_penalty.item(),
            #         torch.mean(critic_clean),
            #         torch.mean(critic_denoised).item(), torch.mean(critic_denoised_retry).item()))
            #
            if global_steps % args.save_every == 0:
                print('Saving Generator and Discriminator Models ')
            #     torch.save(netD, pjoin(args.save_models, 'SEGAN_G_{}.pkl'.format(global_steps)))
            #     torch.save(netG, pjoin(args.save_models, 'SEGAN_D_{}.pkl'.format(global_steps)))
            #     # monitor training
            #     random_z_ = to_device(torch.rand(args.monitor_bs, 1024, 8), device)

                random_z_ = random_z
                gen_ = netG(random_z_, monitor_batch_noisy).detach()
                # error_left = L2_criteria(gen_, monitor_batch_noisy - monitor_batch_clean)
                error_left = L2_criteria(gen_, monitor_batch_clean)
                error_rm = L2_criteria(gen_, monitor_batch_noisy)
                # denoised_ = (monitor_batch_noisy - gen_)
                denoised_ = gen_
                noise_ = monitor_batch_noisy - denoised_
                # gen_ = inv_preemphais_filter(gen_.detach().cpu().numpy())
                noise_ = inv_preemphais_filter(noise_.detach().cpu().numpy())
                denoised_ = inv_preemphais_filter(denoised_.detach().cpu().numpy())
                print('Training Error after {} steps: {}, {}'.format(global_steps, error_left.item(), error_rm.item()))
                print('----------------------------------------------------')

                for i in range(args.monitor_bs):
                    noise_path = pjoin(args.monitor_seen_wav, 'noise_{}_{}.wav'.format(i, global_steps))
                    denoised_path = pjoin(args.monitor_seen_wav, 'denoised_{}_{}.wav'.format(i, global_steps))
                    wavwrite(noise_path, args.fs, noise_[i].T)
                    wavwrite(denoised_path, args.fs, denoised_[i].T)

                # test samples monitor
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

                for i in range(20):
                    noise_path = pjoin(args.monitor_unseen_wav, 'noise_{}_{}.wav'.format(i, global_steps))
                    denoised_path = pjoin(args.monitor_unseen_wav, 'denoised_{}_{}.wav'.format(i, global_steps))
                    wavwrite(noise_path, args.fs, noise_te[i].T)
                    wavwrite(denoised_path, args.fs, denoised_te[i].T)

            if global_steps % (args.save_every * 5) == 0:
                torch.save(netG, pjoin(args.save_models, 'netG_{}.pkl'.format(global_steps)))
