import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle as p
import numpy as np
from os.path import join as pjoin
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir')
parser.add_argument('--data_pickle', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/segan_list.pkl')
parser.add_argument('--bs', type=int, default=3, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
args = parser.parse_args()


class AudioSlicesETL(Dataset):
    def __init__(self, audioslices_pkls=None, audioslices_dir=None):
        if audioslices_pkls is not None:
            with open(audioslices_pkls, 'rb') as rf:
                self.audioslices = p.load(rf)
        else:
            if audioslices_dir is not None:
                self.audioslices = []
                for root, dirs, files in os.walk(audioslices_dir):
                    for file in files:
                        self.audioslices.append(pjoin(root, file))

    def __getitem__(self, idx):
        audioslice_path = self.audioslices[idx]
        pair = np.load(audioslice_path)
        return pair

    def __len__(self):
        return len(self.audioslices)

    def get_reference(self, bs):
        rand_bs = np.random.choice(len(self.audioslices), bs)
        ref_batch = torch.from_numpy(np.stack([np.load(self.audioslices[rand_num]) for rand_num in rand_bs]))
        return ref_batch

    def monitor_eval_training(self, num_examples):
        select_audioslices = np.random.choice(self.audioslices, num_examples, replace=False)
        test_audios = np.stack([np.load(audioslice) for audioslice in select_audioslices])
        test_clean = test_audios[:, 0].reshape(num_examples, 1, 16384)
        test_noisy = test_audios[:, 1].reshape(num_examples, 1, 16384)
        test_clean, test_noisy = [torch.from_numpy(stream) for stream in [test_clean, test_noisy]]
        return test_clean, test_noisy, select_audioslices


if __name__ == '__main__':
    data_tr = AudioSlicesETL(args.data_pickle)
    ref_batch = data_tr.get_reference(args.bs)
    monitor_batch = data_tr.monitor_eval_training(10)
    print('------------------- reference batch ---------------------')
    print(ref_batch[0].shape)
    print('------------------- monitor batch ---------------------')
    print('monitor_clean')
    print(monitor_batch[0][0].shape)
    print('monitor_noisy')
    print(monitor_batch[1][0].shape)

    dl_tr = DataLoader(data_tr, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for idx, batch in enumerate(dl_tr):
        clean, noisy = batch[:, 0], batch[:, 1]  # bs x 2 x signal_len  => [clean, noisy]
