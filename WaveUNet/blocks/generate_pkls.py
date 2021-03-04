import os
from os.path import join as pjoin
import pickle as p


def generate_pkls(audioslices_dir, pkl_file):
    audioslices = []
    for root, dirs, files in os.walk(audioslices_dir):
        for file in files:
            if file.endswith('.npy'):
                audioslices.append(pjoin(root, file))

    with open(pkl_file, 'wb') as wf:
        p.dump(audioslices, wf)


if __name__ == '__main__':
    audioslices_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir'
    audioslices_pkls = '/nas/staff/data_work/Sure/Edinburg_Speech/segan_list.pkl'
    generate_pkls(audioslices_dir, audioslices_pkls)
