import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from os.path import join as pjoin
from data_processing.scaling_spectrum import inverse_in_real_scale, inverse_in_imag_scale, inverse_out_real_scale, inverse_out_imag_scale


feature_maps_dir = '/home/user/on_gpu/CUNet/save_feature_maps'
epoch = 60610
clean_real = np.load(pjoin(feature_maps_dir, 'clean_real_{}.npy'.format(epoch)))
clean_imag = np.load(pjoin(feature_maps_dir, 'clean_imag_{}.npy'.format(epoch)))
noisy_real = np.load(pjoin(feature_maps_dir, 'noisy_real_{}.npy'.format(epoch)))
noisy_imag = np.load(pjoin(feature_maps_dir, 'noisy_imag_{}.npy'.format(epoch)))
dense_real = np.load(pjoin(feature_maps_dir, 'dense_real_{}.npy'.format(epoch)))
dense_imag = np.load(pjoin(feature_maps_dir, 'dense_imag_{}.npy'.format(epoch)))
clean = np.stack([clean_real, clean_imag], axis=-1)
dense = np.stack([dense_real, dense_imag], axis=-1)

clean_real_recover = inverse_out_real_scale(clean_real)
clean_imag_recover = inverse_out_imag_scale(clean_imag)
noisy_real_recover = inverse_in_real_scale(noisy_real)
noisy_imag_recover = inverse_in_imag_scale(noisy_imag)
dense_real_recover = inverse_out_real_scale(dense_real)
dense_imag_recover = inverse_out_imag_scale(dense_imag)

max_value = max(np.max(clean_real_recover), np.max(clean_imag_recover), np.max(noisy_real_recover), np.max(noisy_imag_recover), np.max(dense_real_recover), np.max(dense_imag_recover))
min_value = max(np.min(clean_real_recover), np.min(clean_imag_recover), np.min(noisy_real_recover), np.min(noisy_imag_recover), np.min(dense_real_recover), np.min(dense_imag_recover))
criterion = nn.MSELoss()

clean_real_tc, clean_imag_tc, dense_real_tc, dense_imag_tc, clean_tc, dense_tc = \
    [torch.from_numpy(e) for e in [clean_real, clean_imag, dense_real, dense_imag, clean, dense]]
print(criterion(clean_real_tc, dense_real_tc))
print(criterion(clean_imag_tc, dense_imag_tc))
print(criterion(clean_tc, dense_tc))

# noisy_real, noisy_imag, clean_real, clean_imag, dense_real, dense_imag = [np.log(np.abs(e)) for e in [noisy_real, noisy_imag, clean_real, clean_imag, dense_real, dense_imag]]
max_value_real = max(np.max(clean_real), np.max(noisy_real), np.max(dense_real))
min_value_real = max(np.min(clean_real), np.min(noisy_real), np.min(dense_real))
max_value_imag = max(np.max(clean_imag), np.max(noisy_imag), np.max(dense_imag))
min_value_imag = max(np.min(clean_imag), np.min(noisy_imag), np.min(dense_imag))


plt.figure(1)
plt.subplot(321)
plt.pcolormesh(range(5), range(201), noisy_real_recover, shading='auto')
plt.colorbar()
plt.subplot(322)
plt.pcolormesh(range(5), range(201), noisy_imag_recover, shading='auto')
plt.colorbar()
plt.subplot(323)
plt.pcolormesh(range(5), range(201), clean_real_recover, shading='auto', vmax=max(np.max(clean_real_recover), np.max(dense_real_recover)), vmin=min(np.min(clean_real_recover), np.min(dense_real_recover)))
plt.colorbar()
plt.subplot(324)
plt.pcolormesh(range(5), range(201), clean_imag_recover, shading='auto', vmax=max(np.max(clean_imag_recover), np.max(dense_imag_recover)), vmin=min(np.min(clean_imag_recover), np.min(dense_imag_recover)))
plt.colorbar()
plt.subplot(325)
plt.pcolormesh(range(5), range(201), dense_real_recover, shading='auto', vmax=max(np.max(clean_real_recover), np.max(dense_real_recover)), vmin=min(np.min(clean_real_recover), np.min(dense_real_recover)))
plt.colorbar()
plt.subplot(326)
plt.pcolormesh(range(5), range(201), dense_imag_recover, shading='auto', vmax=max(np.max(clean_imag_recover), np.max(dense_imag_recover)), vmin=min(np.min(clean_imag_recover), np.min(dense_imag_recover)))
plt.colorbar()


plt.figure(2)
plt.subplot(321)
plt.pcolormesh(range(5), range(201), noisy_real, shading='auto')
plt.colorbar()
plt.subplot(322)
plt.pcolormesh(range(5), range(201), noisy_imag, shading='auto')
plt.colorbar()
plt.subplot(323)
plt.pcolormesh(range(5), range(201), clean_real, shading='auto', vmax=max(np.max(clean_real), np.max(dense_real)), vmin=min(np.min(clean_real), np.min(dense_real)))
plt.colorbar()
plt.subplot(324)
plt.pcolormesh(range(5), range(201), clean_imag, shading='auto', vmax=max(np.max(clean_imag), np.max(dense_imag)), vmin=min(np.min(clean_imag), np.min(dense_imag)))
plt.colorbar()
plt.subplot(325)
plt.pcolormesh(range(5), range(201), dense_real, shading='auto', vmax=max(np.max(clean_real), np.max(dense_real)), vmin=min(np.min(clean_real), np.min(dense_real)))
plt.colorbar()
plt.subplot(326)
plt.pcolormesh(range(5), range(201), dense_imag, shading='auto', vmax=max(np.max(clean_imag), np.max(dense_imag)), vmin=min(np.min(clean_imag), np.min(dense_imag)))
plt.colorbar()
plt.show()
