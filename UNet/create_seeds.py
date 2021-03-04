import os
from os.path import join as pjoin
import pickle as p

"""
Directories containing audio or/and video files should be organised as follows:
Data - train - audio
             - video
       valid - audio
             - video
       test  - audio
             - video

               each A/V folder contains audio/video files.

Expected:
       pkls - train.pkl
            - valid.pkl
            - test.pkl

"""


def create_filelist(dir_train, dir_valid, dir_test, dir_pkl):
    """

    :param dir_train: directory for training files
    :param dir_valid: directory for validation files
    :param dir_test: directory for test files
    :param dir_pkl: directory to save pkl filelist
    :return: **
    """

    if 'video' in os.listdir(dir_train):
        additional_input = True
    else:
        additional_input = False

    list_train, list_valid, list_test = [], [], []
    lists_partition = [list_train, list_valid, list_test]
    pkl_names = ['train.pkl', 'valid.pkl', 'test.pkl']
    for idx_dir, dir_split in enumerate([dir_train, dir_valid, dir_test]):
        if additional_input:
            dir_split = pjoin(dir_split, 'audio')

        for root, dirs, files in os.walk(dir_split):
            for file in files:
                if file.endswith('.wav'):
                    path_file = pjoin(root, file)
                    # path_file = [path_file, path_file]
                    path_file = [path_file]
                    if additional_input:
                        path_add_file = pjoin(root.replace('audio', 'video'), file.replace('.wav', '.mpg'))
                        path_file.append(path_add_file)
                    lists_partition[idx_dir].append(path_file)
                    print('Filepath {} added'.format(path_file))

    for idx_pkl, pkl_name in enumerate(pkl_names):
        pkl_path = pjoin(dir_pkl, pkl_name)
        with open(pkl_path, 'wb') as pkl_file:
            p.dump(lists_partition[idx_pkl], pkl_file)


def visualise_filelist(dir_pkl):
    for pkl in os.listdir(dir_pkl):
        if not pkl.startswith('.'):
            with open(pjoin(dir_pkl, pkl), 'rb') as pkl_file:
                filelist = p.load(pkl_file)
                print('-----------------------------------')
                print('Information of Filelist {}'.format(pkl))
                print('- Number of files : {}'.format(len(filelist)))
                if len(filelist):
                    print('- First file : {}'.format(filelist[0]))
                    print('- Last file  : {}'.format(filelist[-1]))
                print()


if __name__ == '__main__':
    dir_voice_ready = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech'
    dir_noise_ready = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise'
    for dir_ready in [dir_voice_ready, dir_noise_ready]:
        dir_train = pjoin(dir_ready, 'train')
        dir_valid = pjoin(dir_ready, 'valid')
        dir_test = pjoin(dir_ready, 'test')
        dir_pkl = pjoin(dir_ready, 'pkls')
        create_filelist(dir_train, dir_valid, dir_test, dir_pkl)
        visualise_filelist(dir_pkl)
