import os
from os.path import join as pjoin
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--processed_data_dir', type=str,
                    default='/nas/staff/data_work/Sure/Edinburg_Speech/processed_data_dir')
args = parser.parse_args()


def compute_max_min(data_dir):
    max_values = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npy'):
                content = np.load(pjoin(root, filename))
                max_values.append(np.max(np.abs(content)))

    return np.max(max_values)


if __name__ == '__main__':
    compute_max_min(args.processed_data_dir)
