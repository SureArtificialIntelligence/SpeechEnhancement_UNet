import torch
import torch.nn as nn
from complex_blocks.cnnlayer_complex import CNN2dLayer as EncoderLayer, CNNTranspose2dLayer as DecoderLayer
from complex_blocks.conv2d_complex import Conv2dComplex
from complex_blocks.relu_complex import TanhComplex
from pars_complexUNet_light import pars_mode1 as pars


class Encoder(nn.Module):
    def __init__(self, pars):
        super(Encoder, self).__init__()
        num_lys = len(pars)
        self.layers = nn.ModuleList([EncoderLayer(pars['layer_{}'.format(i)]) for i in range(1, num_lys + 1)])

    def forward(self, noisy):
        noisy_lys = []
        for ly in self.layers:
            noisy, out_conv = ly(noisy)
            noisy_lys.append(noisy)
        return noisy, noisy_lys[:-1]


class Decoder(nn.Module):
    def __init__(self, pars):
        super(Decoder, self).__init__()
        num_lys = len(pars)
        self.layers = nn.ModuleList([DecoderLayer(pars['layer_{}'.format(i)]) for i in range(1, num_lys + 1)])
        # self.conv = Conv2dComplex(pars[''])  # pars['layer_{}'.format(num_lys)]['tconv']['out_channels'], 1, 1


    def forward(self, x, skips):
        for idx in range(len(self.layers)-1):
            x = self.layers[idx](x, skips[-(idx+1)])
        x = self.layers[-1](x)
        return x


class ComplexUNet(nn.Module):
    def __init__(self, pars):
        super(ComplexUNet, self).__init__()
        self.encoder = Encoder(pars['encoder'])
        self.decoder = Decoder(pars['decoder'])
        self.init_weights()

        self.transitionCNN = Conv2dComplex(pars['trans_layer'])
        self.tanh = TanhComplex()

    def forward(self, noisy, out_format='mask'):
        compress, skips = self.encoder(noisy)
        denoised = self.decoder(compress, skips)
        mask = self.transitionCNN(denoised)
        return self.tanh(mask)

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)


if __name__ == '__main__':
    complexUnet = ComplexUNet(pars)
    print(complexUnet)
    dummy_input = torch.rand((3, 2, 1, 201, 11))
    output = complexUnet(dummy_input)
    print(output.shape)
