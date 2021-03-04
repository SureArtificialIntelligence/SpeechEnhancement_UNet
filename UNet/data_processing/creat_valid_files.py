import os
import numpy as np
import pickle as p
from os.path import join as pjoin
import random
from shutil import copyfile

voice_dir = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech'
noise_dir = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise'

voice_train = pjoin(voice_dir, 'train')
voice_valid = pjoin(voice_dir, 'valid')
voice_test = pjoin(voice_dir, 'test')
noise_train = pjoin(noise_dir, 'train')
noise_valid = pjoin(noise_dir, 'valid')
noise_test = pjoin(noise_dir, 'test')

train_filenames = os.listdir(voice_train)
test_filenames = os.listdir(voice_test)

train_filenames_sample = random.sample(train_filenames, 10)
test_filenames_sample = random.sample(test_filenames, 10)

for train_filename in train_filenames_sample:
    train_voice_path = pjoin(voice_train, train_filename)
    valid_voice_path = pjoin(voice_valid, train_filename)
    train_noise_path = pjoin(noise_train, train_filename)
    valid_noise_path = pjoin(noise_valid, train_filename)

    copyfile(train_voice_path, valid_voice_path)
    copyfile(train_noise_path, valid_noise_path)


for test_filename in test_filenames_sample:
    test_voice_path = pjoin(voice_test, test_filename)
    valid_voice_path = pjoin(voice_valid, test_filename)
    test_noise_path = pjoin(noise_test, test_filename)
    valid_noise_path = pjoin(noise_valid, test_filename)

    copyfile(test_voice_path, valid_voice_path)
    copyfile(test_noise_path, valid_noise_path)

filenames = []
filenames.extend(train_filenames_sample)
filenames.extend(test_filenames_sample)
print(filenames)
with open('./valid_list.txt', 'w') as f:
    for filename in filenames:
        f.write('{}\n'.format(filename))
