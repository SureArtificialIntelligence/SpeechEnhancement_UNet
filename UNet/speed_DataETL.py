import time
from DataETL import DataETL, DataLoader
signal_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/train.pkl'
noise_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/train.pkl'

data_etl = DataETL('train', signal_filelist_path=signal_seeds, noise_filelist_path=noise_seeds,
                   feature='log-magnitude', slice_win=5, target_scope=2, num_segments=5)
data_loader = DataLoader(data_etl, shuffle=True, batch_size=20, num_workers=16)

current_time = time.clock()
for idx_batch, batch in enumerate(data_loader):
    update_time = time.clock()
    duration = update_time - current_time
    print('No.{}, Loading Time: {}'.format(idx_batch, duration))
    print('Elements in one batch: {}'.format(list(batch.keys())))
    current_time = time.clock()
