import torch
import torch.nn as nn


class Conv2dComplex(nn.Module):
    def __init__(self, pars):
        super(Conv2dComplex, self).__init__()
        self.conv_real = nn.Conv2d(**pars)
        self.conv_imag = nn.Conv2d(**pars)

    def forward(self, x):
        # x: bs x [real, imag] x ch x F x T
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.conv_real(in_real) - self.conv_imag(in_imag)
        out_imag = self.conv_imag(in_real) + self.conv_real(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out


class ConvTranspose2dComplex(nn.Module):
    def __init__(self, pars):
        super(ConvTranspose2dComplex, self).__init__()
        self.tconv_real = nn.ConvTranspose2d(**pars)
        self.tconv_imag = nn.ConvTranspose2d(**pars)

    def forward(self, x):
        in_real, in_imag = x[:, 0], x[:, 1]
        out_real = self.tconv_real(in_real) - self.tconv_imag(in_imag)
        out_imag = self.tconv_imag(in_real) + self.tconv_real(in_imag)
        out = torch.stack([out_real, out_imag], dim=1)
        return out


if __name__ == '__main__':
    pars_conv = {
        'in_channels': 8,
        'out_channels': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    }
    complex_conv2d = Conv2dComplex(pars_conv)
    dummy_input = torch.rand((3, 2, 8, 100, 201))
    print(dummy_input.shape)
    output = complex_conv2d(dummy_input)
    print(output.shape)

    pars_tconv = {
        'in_channels': 32,
        'out_channels': 8,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    }
    complex_tconv2d = ConvTranspose2dComplex(pars_tconv)
    output = complex_tconv2d(output)
    print(output.shape)
