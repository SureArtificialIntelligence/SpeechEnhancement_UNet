import threading
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin


GLOBAL_MIN = 2


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def monitor_training():
    threading.Timer(60.0, monitor_training).start()
    print_out_files = []
    # print_out_file1 = pjoin('/home/user/on_gpu/slurm_info/UNET', '101115')
    # print_out_files.append(print_out_file1)
    # print_out_file2 = pjoin('/home/user/on_gpu/slurm_info/UNET', '101119')
    # print_out_files.append(print_out_file2)
    print_out_file3 = pjoin('/home/user/on_gpu/slurm_info/UNET', '101253')
    print_out_files.append(print_out_file3)

    plt.figure()
    for print_out_file in print_out_files:
        n_batches, loss_gs, n_batches_v, loss_vs = [], [], [], []
        with open(print_out_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('['):
                    elements = line.split(' ')
                    n_batch = int(elements[1][1:])
                    # n_batch = int(elements[2][:-1])
                    loss_g = float(elements[-1][:-1])

                    n_batches.append(n_batch)
                    loss_gs.append(loss_g)

                elif line.startswith('=>'):
                    n_batches_v.append(n_batches[-1])
                    elements = line.split(' ')
                    loss_v = float(elements[-1][:-1])
                    loss_vs.append(loss_v)

        loss_gs_ma = moving_average(loss_gs, 1)
        # loss_vs_ma = moving_average(loss_vs, 1)
        loss_min = min(loss_gs)
        idx_min = loss_gs.index(loss_min)
        idx = n_batches[idx_min]

        # global GLOBAL_MIN
        # if loss_min < GLOBAL_MIN:
        # GLOBAL_MIN = loss_min

        plt.plot(n_batches, loss_gs, 'k')
        plt.plot(n_batches, loss_gs_ma, 'r')
        # plt.plot(n_batches_v, loss_vs_ma, 'b')
        plt.plot(idx, loss_min, 'o')

    plt.title('Current minimum loss: {:.5f}'.format(loss_min))
    plt.show()


monitor_training()
