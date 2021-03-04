import os
import numpy as np
from os.path import join as pjoin

src_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_test'
dst_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir_test_shortversion'

num_files = 20

filenames = os.listdir(src_dir)
slt_filenames = np.random.choice(filenames, num_files, replace=False)
[os.system('cp {} {}'.format(pjoin(src_dir, fn), pjoin(dst_dir, fn))) for fn in slt_filenames]
