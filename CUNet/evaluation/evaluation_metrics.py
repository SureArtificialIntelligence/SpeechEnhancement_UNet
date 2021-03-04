# Audio enhancement evaluation metrics

# PESQ seems complicated here.

import numpy as np
import museval
import os
from os.path import join as pjoin
from pyroomacoustics import metrics
from scipy.io.wavfile import read as wavread
from scipy.signal import stft, istft
from pystoi.stoi import stoi


def compute_LSD(reference, estimated):
    fs_ref, ref_wavform = wavread(reference)
    fs_est, est_wavform = wavread(estimated)
    assert fs_est == fs_ref
    f, t, ref_stft = stft(ref_wavform, fs_est, nperseg=400, noverlap=160, nfft=400)
    f, t, est_stft = stft(est_wavform, fs_est, nperseg=400, noverlap=160, nfft=400)
    lsd = np.mean(np.sqrt(np.mean(np.square(10*np.log10(abs(ref_stft) / abs(est_stft))), axis=0)))
    return lsd


def compute_SDR(reference, estimated):
    fs_ref, ref_wavform = wavread(reference)
    fs_est, est_wavform = wavread(estimated)
    sdr, _, _, _ = museval.metrics.bss_eval_sources(ref_wavform, est_wavform)
    return sdr


def compute_STOI(reference, estimated):
    fs_ref, ref_wavform = wavread(reference)
    fs_est, est_wavform = wavread(estimated)
    STOI = stoi(ref_wavform, est_wavform, fs_est)
    return STOI


def compute_PESQ(reference, estimated):
    target = reference[:-len(reference.split('/')[-1])-len(reference.split('/')[-2])-1] + 'temp/' + 'target.wav'
    degrade = estimated[:-len(estimated.split('/')[-1])-len(estimated.split('/')[-2])-1] + 'temp/' + 'degrade.wav'
    # degrade = target
    change_name1 = 'cp {} {}'.format(reference, target)
    change_name2 = 'cp {} {}'.format(estimated, degrade)
    os.system(change_name1)
    os.system(change_name2)
    pesq = metrics.pesq(target, degrade, 16000, bin=b'/home/user/pesq/Software/P862_annex_A_2005_CD/source/PESQ')
    os.system('rm -rf {}'.format(target))
    os.system('rm -rf {}'.format(degrade))
    return pesq


def compute_CD(reference, estimated):
    fs_ref, ref_wavform = wavread(reference)
    fs_est, est_wavform = wavread(estimated)
    # ref_cepstrum = librosa.feature.mfcc(ref_wavform, sr=fs_ref, n_mfcc=8)
    # est_cepstrum = librosa.feature.mfcc(est_wavform, sr=fs_est, n_mfcc=8)
    assert fs_est == fs_ref
    f, t, ref_stft = stft(ref_wavform, fs_est, nperseg=400, noverlap=160, nfft=400)
    f, t, est_stft = stft(est_wavform, fs_est, nperseg=400, noverlap=160, nfft=400)
    ref_cepstrum = np.real(istft(np.log(abs(ref_stft)), fs_ref, nperseg=400, noverlap=160, nfft=400))
    est_cepstrum = np.real(istft(np.log(abs(est_stft)), fs_ref, nperseg=400, noverlap=160, nfft=400))
    mcd = np.sqrt(np.sum((ref_cepstrum - est_cepstrum) ** 2, axis=0)).mean()
    # mcd = metrics.melcd(est_cepstrum, ref_cepstrum)
    return mcd


def compute_segSNR(reference, estimated):
    seg_SNRs = []
    fs_ref, ref_wavform = wavread(reference)
    fs_est, est_wavform = wavread(estimated)
    num_segs = int((len(ref_wavform) - 400) / 160)
    for ii in range(num_segs):
        ref_seg = ref_wavform[ii*160 : ii*160+400]
        est_seg = est_wavform[ii*160 : ii*160+400]
        if (ref_seg ** 2).sum() > 0 and ((ref_seg - est_seg) ** 2).sum() > 0:
            seg_SNR = np.mean(10 * np.log10((ref_seg ** 2).sum() / ((ref_seg - est_seg) ** 2).sum()))
            seg_SNRs.append(seg_SNR)
    return np.mean(seg_SNRs)


if __name__ == '__main__':
    root_dir = '/nas/staff/data_work/Sure/GPU_wav_dump/SECUNet/wi_wsdr/wav_test'
    ids = []
    for filename in os.listdir(root_dir):
        if filename.startswith('denoised_'):
            ids.append(filename[9:-4])

    orig_segSNRs, new_segSNRs, orig_CDs, new_CDs, orig_SDRs, new_SDRs, orig_STOIs, new_STOIs, orig_LSDs, new_LSDs = [], [], [], [], [], [], [], [], [], []
    for idx, wave_id in enumerate(ids):
        # wave_id = 'p232_005.wav_10'
        print('Processing {}/{}'.format(idx, len(ids)))
        clean = pjoin(root_dir, 'clean_'+wave_id+'.wav')
        denoised = pjoin(root_dir, 'denoised_'+wave_id+'.wav')
        noisy = pjoin(root_dir, 'noisy_'+wave_id+'.wav')

        orig_segSNR, new_segSNR = compute_segSNR(clean, noisy), compute_segSNR(clean, denoised)
        orig_segSNRs.append(orig_segSNR)
        new_segSNRs.append(new_segSNR)

        orig_CD, new_CD = compute_CD(clean, noisy), compute_CD(clean, denoised)
        orig_CDs.append(orig_CD)
        new_CDs.append(new_CD)

        orig_SDR, new_SDR = compute_SDR(clean, noisy), compute_SDR(clean, denoised)
        orig_SDRs.append(orig_SDR)
        new_SDRs.append(new_SDR)

        orig_stoi, new_stoi = compute_STOI(clean, noisy), compute_STOI(clean, denoised)
        orig_STOIs.append(orig_stoi)
        new_STOIs.append(new_stoi)

        orig_lsd, new_lsd = compute_LSD(clean, noisy), compute_LSD(clean, denoised)
        orig_LSDs.append(orig_lsd)
        new_LSDs.append(new_lsd)

    print('segemental SNR: {} -> {}'.format(np.mean(orig_segSNRs), np.mean(new_segSNRs)))
    print('Ceptral distortion: {} -> {}'.format(np.mean(orig_CDs), np.mean(new_CDs)))
    print('SDR: {} -> {}'.format(np.mean(orig_SDRs), np.mean(new_SDRs)))
    print('STOI: {} -> {}'.format(np.mean(orig_STOIs), np.mean(new_STOIs)))
    print('LSD: {} -> {}'.format(np.mean(orig_LSDs), np.mean(new_LSDs)))

