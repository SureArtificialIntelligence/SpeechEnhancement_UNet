import matplotlib.pyplot as plt
from os.path import join as pjoin
print_out_file = pjoin('/home/user/on_gpu/slurm_info/SE_UNet/CUN', '105376_0')

n_batches, loss_gs, loss_rs, loss_is = [], [], [], []
with open(print_out_file) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('['):
            elements = line.split(' ')
            print(elements)
            n_batch = int(elements[1][:-1])
            loss_g = float(elements[3][:-1])
            loss_r = float(elements[6][:-1])
            loss_i = float(elements[-1][:-1])

            n_batches.append(n_batch)
            loss_gs.append(loss_g)
            loss_rs.append(loss_r)
            loss_is.append(loss_i)

plt.figure()
plt.plot(n_batches, loss_gs, 'k')
plt.plot(n_batches, loss_rs, 'r')
plt.plot(n_batches, loss_is, 'b')
# plt.ylim([0, 0.002])
plt.show()
