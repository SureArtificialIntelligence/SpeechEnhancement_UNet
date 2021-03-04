import os
from os.path import join as pjoin
import pickle as p


def create_pkls(src_dir, save_to):
    filenames = []
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith('.npy'):
                filepath = pjoin(root, filename)
                print('Processing {}'.format(filepath))
                filenames.append(filepath)

    with open(save_to, 'wb') as pkl_file:
        p.dump(filenames, pkl_file)


def check_pkls(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        seeds = p.load(pkl_file)
    for seed in seeds:
        seed.split('/')[-1].split('.')
    print(seeds)
    print(len(seeds))


if __name__ == '__main__':
    src_folder = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_MagPhase'
    pkls_save2 = '/nas/staff/data_work/Sure/Edinburg_Speech/magphase_train.pkl'
    # create_pkls(src_folder, pkls_save2)
    check_pkls(pkls_save2)
