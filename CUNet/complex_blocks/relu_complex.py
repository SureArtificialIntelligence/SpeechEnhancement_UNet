import torch
import torch.nn as nn


class ReLUComplex(nn.Module):
    def __init__(self, pars):
        super(ReLUComplex, self).__init__()
        self.act_real = nn.ReLU(**pars)
        self.act_imag = nn.ReLU(**pars)

    def forward(self, x):
        # x: bs x [real, imag] x ch x F x T
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.act_real(in_real)
        out_imag = self.act_imag(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out


class LeakyReLUComplex(nn.Module):
    def __init__(self, pars):
        super(LeakyReLUComplex, self).__init__()
        self.act_real = nn.LeakyReLU(**pars)
        self.act_imag = nn.LeakyReLU(**pars)

    def forward(self, x):
        # x: bs x [re, im] x ch x F x T
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.act_real(in_real)
        out_imag = self.act_imag(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out


class TanhComplex(nn.Module):
    def __init__(self):
        super(TanhComplex, self).__init__()
        self.act_real = nn.Tanh()
        self.act_imag = nn.Tanh()

    def forward(self, x):
        # x: bs x [re, im] x ch x F x T
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.act_real(in_real)
        out_imag = self.act_imag(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out