import torch
import torch.nn as nn
# from blocks.virtual_batchnorm import VirtualBatchNorm1d
from blocks.VirtualBN1D import VirtualBN1D
from nn_pars import pars_generator, pars_discriminator


class GenEncoderLayer(nn.Module):
    def __init__(self, pars):
        super(GenEncoderLayer, self).__init__()
        self.conv = nn.Conv1d(**pars['conv'])
        # self.bn = VirtualBatchNorm1d(**pars['vbn'])
        self.bn = nn.BatchNorm1d(**pars['norm'])
        self.act = nn.PReLU()

    def forward(self, x):
        # out_conv = self.bn(self.conv(x))
        out_conv = self.conv(x)
        return self.act(out_conv), out_conv


class GenDecoderLayer(nn.Module):
    def __init__(self, pars):
        super(GenDecoderLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(**pars['conv'])
        # self.bn = VirtualBatchNorm1d(**pars['vbn'])
        self.bn = nn.BatchNorm1d(**pars['norm'])
        self.act = nn.PReLU()
        self.act_final = nn.Tanh()

    def forward(self, x, skip=None):
        """

        :param x: bs x ch x len
        :param skip: bs x ch x len
        :return:
        """
        out = self.conv(x)
        if skip is not None:
            out = torch.cat([out, skip], 1)
            out = self.bn(out)
            out = self.act(out)
        else:
            out = self.act_final(out)
            # out = out
        return out


class GenEncoder(nn.Module):
    def __init__(self, pars):
        super(GenEncoder, self).__init__()
        self.layers = nn.ModuleList([GenEncoderLayer(pars['layer{}'.format(i)])
                                     for i in range(1, len(pars) + 1)])
        # self.encoder = nn.Sequential(*layers)

    def forward(self, noisy):
        noisy_lys = []
        for ly in self.layers:
            noisy, out_conv = ly(noisy)
            noisy_lys.append(out_conv)
        return noisy, noisy_lys[:-1]


class GenDecoder(nn.Module):
    def __init__(self, pars):
        super(GenDecoder, self).__init__()
        self.layers = nn.ModuleList([GenDecoderLayer(pars['layer{}'.format(i)])
                                     for i in range(1, len(pars) + 1)])

    def forward(self, x, skips):
        for idx in range(len(self.layers)-1):
            x = self.layers[idx](x, skips[-(idx+1)])
        x = self.layers[-1](x)
        return x


class Generator(nn.Module):
    def __init__(self, pars):
        super(Generator, self).__init__()
        self.encoder = GenEncoder(pars['encoder'])
        self.decoder = GenDecoder(pars['decoder'])

        self.init_weights()

    def forward(self, z, noisy):
        """

        :param z: random vector
        :param noisy: noisy signal [bs x len]
        :return: [bs x ch x len_]
        """
        compress, skips = self.encoder(noisy)  # noisy: bs x ch x len
        # compress_z = torch.cat([compress, z], 1)
        compress_z = compress
        denoised = self.decoder(compress_z, skips)
        return denoised

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)


class DiscriminatorLayer(nn.Module):
    def __init__(self, pars):
        super(DiscriminatorLayer, self).__init__()
        self.conv = nn.Conv1d(**pars['conv'])
        self.vbn = VirtualBN1D(**pars['vbn'])
        self.bn = nn.BatchNorm1d(**pars['vbn'])
        self.act = nn.LeakyReLU(**pars['act'])

    def forward(self, x, ref_x=None):
        ref_out = ref_x
        if ref_x is not None:
            ref_out, ref_mean, ref_mean_sq = self.vbn.prepare(self.conv(ref_out))
            out = self.vbn(self.conv(x), ref_mean, ref_mean_sq)
        else:
            out = self.bn(self.conv(x))
        return self.act(out), self.act(ref_out)


class Discriminator(nn.Module):
    def __init__(self, pars):
        super(Discriminator, self).__init__()
        # num_layers = len([par for par in pars if par.startswith('layer')])
        self.layers = nn.ModuleList([DiscriminatorLayer(pars['layers']['layer{}'.format(i)])
                                     for i in range(1, len(pars['layers']) + 1)])
        self.squeeze_conv = nn.Conv1d(**pars['squeeze_conv'])
        self.lrelu = nn.LeakyReLU(**pars['lrelu'])
        self.project = nn.Linear(**pars['project'])

        self.init_weights()

    def forward(self, denoised, noisy, reference):
        out = torch.cat([denoised, noisy], 1)
        ref_out = reference
        for ly in self.layers:
            out, ref_out = ly(out, ref_out)
        out = self.squeeze_conv(out).view(-1, 8)
        out = self.project(out)
        return out

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)


if __name__ == '__main__':
    netG = Generator(pars_generator)
    netD = Discriminator(pars_discriminator)
    print('--------------------------------- Generator ------------------------------')
    print(netG)
    print('-------------------------------- Disciminator ----------------------------')
    print(netD)

    print('--------------------------------- Dummy Input ----------------------------')
    dummy_input = torch.zeros((2, 1, 16384))
    rand_z = torch.rand((2, 1024, 8))
    outG = netG(rand_z, dummy_input)
    print(outG.shape)

    target = torch.ones((2, 1, 16384))
    reference = torch.ones(2, 2, 16384)
    outD = netD(outG, target, reference)
    print(outD)
