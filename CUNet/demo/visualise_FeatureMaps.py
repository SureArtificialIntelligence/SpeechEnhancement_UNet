import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin


feature_maps_dir = './save_feature_maps'
epoch = 21200
clean_real = np.load(pjoin(feature_maps_dir, 'clean_real_{}'.format(epoch)))
clean_imag = np.load(pjoin(feature_maps_dir, 'clean_imag_{}'.format(epoch)))
noisy_real = np.load(pjoin(feature_maps_dir, 'noisy_real_{}'.format(epoch)))
noisy_imag = np.load(pjoin(feature_maps_dir, 'noisy_imag_{}'.format(epoch)))
dense_real = np.load(pjoin(feature_maps_dir, 'dense_real_{}'.format(epoch)))
dense_imag = np.load(pjoin(feature_maps_dir, 'dense_imag_{}'.format(epoch)))

plt.figure()
plt.subplot(321)
plt.pcolormesh(range(5), range(201), noisy_real, shading='auto')
plt.subplot(322)
plt.pcolormesh(range(5), range(201), noisy_imag, shading='auto')
plt.subplot(323)
plt.pcolormesh(range(5), range(201), clean_real, shading='auto')
plt.subplot(324)
plt.pcolormesh(range(5), range(201), clean_imag, shading='auto')
plt.subplot(325)
plt.pcolormesh(range(5), range(201), dense_real, shading='auto')
plt.subplot(326)
plt.pcolormesh(range(5), range(201), dense_imag, shading='auto')
plt.show()