import torch
import torch.nn as nn

Print_shape = True


class CNN2dLayer(nn.Module):
    def __init__(self, pars):
        super(CNN2dLayer, self).__init__()
        self.conv = nn.Conv2d(**pars['conv'])
        self.bn = nn.BatchNorm2d(**pars['bn'])
        self.act = nn.LeakyReLU(**pars['act'])
        # self.act = nn.ReLU()

    def forward(self, x):
        if Print_shape:
            print('encoder')
            print(x.shape)
        # x: bs x [real, imag] x ch x F x T
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)
        if Print_shape:
            print(conv_out.shape)
        return self.act(bn_out), conv_out
        # return self.bn(self.act(conv_out)), conv_out


class CNNTranspose2dLayer(nn.Module):
    def __init__(self, pars):
        super(CNNTranspose2dLayer, self).__init__()
        self.tconv = nn.ConvTranspose2d(**pars['tconv'])
        self.bn = nn.BatchNorm2d(**pars['bn'])
        self.act = nn.LeakyReLU(**pars['act'])
        # self.act = nn.ReLU()

    def forward(self, x, mode, skip=None):
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
            if mode == 0:
                out = torch.cat([out, skip], dim=1)
                out = self.act(self.bn(out))
                # out = self.bn(self.act(out))
            else:
                out = self.act(self.bn(out))
                # out = self.bn(self.act(out))
                out = torch.cat([out, skip], dim=1)
        else:
            out = self.act(self.bn(out))
            # out = out
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
            'inplace': True,
            'negative_slope': 0.01
        }
    }
    pars_tcnn = {
        'tconv': {
            'in_channels': 32,
            'out_channels': 8,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'bn': {
            'num_features': 8
        },
        'act': {
            'inplace': True,
            'negative_slope': 0.01
        }
    }
    cnn2d = CNN2dLayer(pars_cnn)
    tcnn2d = CNNTranspose2dLayer(pars_tcnn)
    dummy_input = torch.rand((3, 8, 100, 201))
    # print(dummy_input.shape)
    out, skip = cnn2d(dummy_input)
    # print(out.shape, skip.shape)
    out = tcnn2d(out, skip)
    # print(out.shape)
