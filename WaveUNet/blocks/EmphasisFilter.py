from scipy import signal
import numpy as np


def normalize_wave_minmax(x):
    return (2./65536.) * (x - 32768.) + 1.


def preemphais_filter(data, emph_coeff=0.95):
    data_norm = normalize_wave_minmax(data)
    data_emph = signal.lfilter([1, -emph_coeff], [1], data_norm)
    return data_emph


def inv_preemphais_filter(data, emph_coeff=0.95):
    return signal.lfilter([1], [1, -emph_coeff], data)


if __name__ == '__main__':
    rand_signal_batch = np.random.randint(low=1, high=10, size=(10, 1, 400))
    reconstructed = inv_preemphais_filter(preemphais_filter(rand_signal_batch))
    assert rand_signal_batch.shape == reconstructed.shape
    print((rand_signal_batch == reconstructed).all())
