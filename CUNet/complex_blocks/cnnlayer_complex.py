import torch
import torch.nn as nn
from complex_blocks.conv2d_complex import Conv2dComplex, ConvTranspose2dComplex
from complex_blocks.batchnorm_complex import BatchNormComplex
from complex_blocks.relu_complex import ReLUComplex, LeakyReLUComplex

Print_shape = False


class CNN2dLayer(nn.Module):
    def __init__(self, pars):
        super(CNN2dLayer, self).__init__()
        self.conv = Conv2dComplex(pars['conv'])
        self.bn = BatchNormComplex(pars['bn'])
        self.act = LeakyReLUComplex(pars['act'])

    def forward(self, x):
        if Print_shape:
            print('encoder')
            print(x.shape)
        # x: bs x [real, imag] x ch x F x T
        conv_out = self.bn(self.conv(x))
        if Print_shape:
            print(conv_out.shape)
        return self.act(conv_out), conv_out


class CNNTranspose2dLayer(nn.Module):
    def __init__(self, pars):
        super(CNNTranspose2dLayer, self).__init__()
        self.tconv = ConvTranspose2dComplex(pars['tconv'])
        self.bn = BatchNormComplex(pars['bn'])
        self.act = LeakyReLUComplex(pars['act'])
        self.act_final = nn.Tanh()

    def forward(self, x, skip=None, mode=1):
        # x: bs x [real, imag] x ch x F x T
        if Print_shape:
            print('decoder')
            print(x.shape)
        out = self.tconv(x)
        if Print_shape:
            print(out.shape)
        if skip is not None:
            if Print_shape:
                print(skip.shape)
            if mode==0:
                out = torch.cat([out, skip], dim=2)
                out = self.act(self.bn(out))
            else:
                out = self.act(self.bn(out))
                out = torch.cat([out, skip], dim=2)
        else:
            # out = self.act_final(out)
            out = self.act(self.bn(out))
        return out


if __name__ == '__main__':
    pars_cnn = {
        'conv': {
            'in_channels': 8,
            'out_channels': 32,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'bn': {
            'num_features': 32
        },
        'act': {
            'inplace': True
        }
    }
    pars_tcnn = {
        'tconv': {
            'in_channels': 32 * 2,
            'out_channels': 8,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'bn': {
            'num_features': 8
        },
        'act': {
            'inplace': True
        }
    }
    cnn2d = CNN2dLayer(pars_cnn)
    tcnn2d = CNNTranspose2dLayer(pars_tcnn)
    dummy_input = torch.rand((3, 2, 1, 201, 11))
    print(dummy_input.shape)
    out, skip = cnn2d(dummy_input)
    print(out.shape, skip.shape)
    out = tcnn2d(out, skip)
    print(out.shape)
