import pickle as p
import os
import torch
import torchaudio


## from txt file
# testset_txt = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/meta/log_testset.txt'
# snr_dict = {}
# with open(testset_txt, 'r') as file:
#     for line in file.readlines():
#         elements = line.split(' ')
#         id, snr = elements[0], float(elements[2][:-1])
#         snr_dict[id] = snr
#
# with open('/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/meta/snr_dict.pkl', 'wb') as p_file:
#     p.dump(snr_dict, p_file)

## from wav file
root_speech = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/test'
root_noise = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/test'
snr_dict = {}
for root, dirs, files in os.walk(root_speech):
    for file in files:
        if file.endswith('.wav'):
            speech_path = os.path.join(root_speech, file)
            noise_path = os.path.join(root_noise, file)
            speech_wav, speech_sr = torchaudio.load(speech_path, normalization=False)
            noise_wav, noise_sr = torchaudio.load(noise_path, normalization=False)
            p_speech = torch.sum(speech_wav ** 2) / speech_wav.size()[1]
            p_noise = torch.sum(noise_wav ** 2) / noise_wav.size()[1]
            snr = 10 * torch.log10(p_speech / p_noise)
            snr_dict[file[:-4]] = snr

with open('/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/meta/snr_dict2.pkl', 'wb') as p_file:
    p.dump(snr_dict, p_file)
