import torch
import torch.nn as nn


class BatchNormComplex(nn.Module):
    def __init__(self, pars):
        super(BatchNormComplex, self).__init__()
        self.bn_real = nn.BatchNorm2d(**pars)
        self.bn_imag = nn.BatchNorm2d(**pars)

    def forward(self, x):
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.bn_real(in_real)
        out_imag = self.bn_real(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out
