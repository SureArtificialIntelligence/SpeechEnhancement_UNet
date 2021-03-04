from scipy.io.wavfile import read as wavread
import pysepm
from tqdm import tqdm
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_wav_dir', type=str,
                    default='/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/CUNet/wSDR/monitor_unseen_wav')
args = parser.parse_args()


def measure_evaluation_metics(clean_path, measure_path, metrics):
    evaluation_metrics = {
        'fwSNRseg': pysepm.fwSNRseg,
        'SNRseg': pysepm.SNRseg,
        'LLR': pysepm.llr,
        'WSS': pysepm.wss,
        'cd': pysepm.cepstrum_distance,
        'stoi': pysepm.stoi,
        'pesq': pysepm.pesq,
        'csii': pysepm.csii,
        'composite': pysepm.composite,  # csig, cbak, covl
        'ncm': pysepm.ncm,
        'srmr': pysepm.srmr,
        'bsd': pysepm.bsd,
    }

    em_funcs = []
    for em in metrics:
        em_funcs.append(evaluation_metrics[em])

    fs, clean_speech = wavread(clean_path)
    fs, measure_speech = wavread(measure_path)

    evaluation_results = [em_func(clean_speech, measure_speech, fs) for em_func in em_funcs]

    assert len(evaluation_results) == len(metrics)

    results = {}
    for idx, em in enumerate(metrics):
        if em == 'composite':
            results['csig'], results['cbak'], results['covl'] = evaluation_results[idx]
        elif em == 'pesq':
            results['pesq'] = evaluation_results[idx][1]
        else:
            results[em] = evaluation_results[idx]

    return results


def merge_multiple_results(results_list):
    keys = results_list[0].keys()
    results = {}
    for key in keys:
        results[key] = [res[key] for res in results_list]
    return results


if __name__ == '__main__':
    considered = ['composite', 'pesq']

    file_id_all = []
    results_original_aggregate = []
    results_enhanced_aggregate = []
    root_wav_dir = args.root_wav_dir

    for root, dirs, files in os.walk(root_wav_dir):
        for file in tqdm(files):
            if file.endswith('.wav') and file.startswith('denoised'):
                try:
                    file_id = file[9:-4]
                    file_id_all.append(file_id)
                    print('Meausuring file: {}'.format(file_id))
                    enhanced_path = os.path.join(root, file)
                    noisy_path = enhanced_path.replace('denoised', 'noisy')
                    clean_path = enhanced_path.replace('denoised', 'clean')

                    results_original = measure_evaluation_metics(clean_path, noisy_path, considered)
                    results_enhanced = measure_evaluation_metics(clean_path, enhanced_path, considered)
                    # print(results_original)
                    # print(results_enhanced)
                    # print('-------------------------------------')
                    results_original_aggregate.append(results_original)
                    results_enhanced_aggregate.append(results_enhanced)

                except:
                    print('Processing Error: {}'.format(file_id))
                    pass

    results_original_all = merge_multiple_results(results_original_aggregate)
    results_enhanced_all = merge_multiple_results(results_enhanced_aggregate)

    if 'composite' in considered:
        considered.remove('composite')
        considered.extend(['csig', 'cbak', 'covl'])
    avg_pesq_original, avg_csig_original, avg_cbak_original, avg_covl_original = [np.mean(results_original_all[em]) for em in considered]
    avg_pesq_enhanced, avg_csig_enhanced, avg_cbak_enhanced, avg_covl_enhanced = [np.mean(results_enhanced_all[em]) for em in considered]

    avg_pesq_original = avg_pesq_original[~np.isnan(avg_pesq_original)]
    avg_pesq_enhanced = avg_pesq_enhanced[~np.isnan(avg_pesq_enhanced)]

    print('------ original => enhanced --------')
    print('csig: {} => {}'.format(avg_csig_original, avg_csig_enhanced))
    print('cbak: {} => {}'.format(avg_cbak_original, avg_cbak_enhanced))
    print('covl: {} => {}'.format(avg_covl_original, avg_covl_enhanced))
    print('pesq: {} => {}'.format(avg_pesq_original, avg_pesq_enhanced))
