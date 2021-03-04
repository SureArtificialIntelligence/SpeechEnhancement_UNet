import numpy as np


def scaling(x, max_v, min_v):
    x_norm = (np.array(x) - min_v - (max_v - min_v) / 2) / ((max_v - min_v) / 2)
    return x_norm


def inverse_scaling(x, max_v, min_v):
    x_invnorm = x * ((max_v - min_v) / 2) + ((max_v - min_v) / 2) + min_v
    return x_invnorm


# in_real_max, in_real_min = 196.0, -200.0
# in_imag_max, in_imag_min = 166.0, -124.0
# out_real_max, out_real_min = 154.0, -138.0
# out_imag_max, out_imag_min = 87.0, -98.0

in_real_max, in_real_min = 196.0, -200.0
in_imag_max, in_imag_min = 196.0, -200.0
out_real_max, out_real_min = 154.0, -138.0
out_imag_max, out_imag_min = 154.0, -138.0

# in_real_max, in_real_min = 196.0, -200.0
# in_imag_max, in_imag_min = 196.0, -200.0
# out_real_max, out_real_min = 196.0, -200.0
# out_imag_max, out_imag_min = 196.0, -200.0


def in_real_scale(x):
    return scaling(x, in_real_max, in_real_min)


def in_imag_scale(x):
    return scaling(x, in_imag_max, in_imag_min)


def out_real_scale(x):
    return scaling(x, out_real_max, out_real_min)


def out_imag_scale(x):
    return scaling(x, out_imag_max, out_imag_min)


def inverse_in_real_scale(x):
    return inverse_scaling(x, in_real_max, in_real_min)


def inverse_in_imag_scale(x):
    return inverse_scaling(x, in_imag_max, in_imag_min)


def inverse_out_real_scale(x):
    return inverse_scaling(x, out_real_max, out_real_min)


def inverse_out_imag_scale(x):
    return inverse_scaling(x, out_imag_max, out_imag_min)


if __name__ == '__main__':
    dummy_x = 2.19088380e-02
    out = in_real_scale(dummy_x)
    out = inverse_in_real_scale(out)
    assert dummy_x == out

    out = in_imag_scale(dummy_x)
    out = inverse_in_imag_scale(out)
    assert dummy_x == out

    out = out_real_scale(dummy_x)
    out = inverse_out_real_scale(out)
    assert dummy_x == out

    out = out_imag_scale(dummy_x)
    out = inverse_out_imag_scale(out)
    assert dummy_x == out
