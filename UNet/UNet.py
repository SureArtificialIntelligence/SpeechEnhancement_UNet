import torch
import torch.nn as nn
from blocks.cnnlayer import CNN2dLayer as EncoderLayer, CNNTranspose2dLayer as DecoderLayer
from pars_UNet_light import pars_mode1 as pars

# mode:0 is consitent with github implementation


class Encoder(nn.Module):
    def __init__(self, pars, mode):
        super(Encoder, self).__init__()
        num_lys = len(pars)
        self.layers = nn.ModuleList([EncoderLayer(pars['layer_{}'.format(i)]) for i in range(1, num_lys + 1)])
        self.mode = mode

    def forward(self, noisy):
        noisy_lys = []
        for ly in self.layers:
            noisy, out_conv = ly(noisy)
            if self.mode == 0:
                noisy_lys.append(out_conv)
            else:
                noisy_lys.append(noisy)
        return noisy, noisy_lys[:-1]


class Decoder(nn.Module):
    def __init__(self, pars, mode):
        super(Decoder, self).__init__()
        num_lys = len(pars)
        self.layers = nn.ModuleList([DecoderLayer(pars['layer_{}'.format(i)]) for i in range(1, num_lys + 1)])
        self.mode = mode
        self.conv = nn.Conv2d(pars['layer_{}'.format(num_lys)]['tconv']['out_channels'], 1, 1)

    def forward(self, x, skips):
        for idx in range(len(self.layers)-1):
            x = self.layers[idx](x, self.mode, skips[-(idx+1)])
        x = self.layers[-1](x, self.mode)
        return x


class UNet(nn.Module):
    def __init__(self, pars):
        super(UNet, self).__init__()
        self.encoder = Encoder(pars['encoder'], pars['mode'])
        self.decoder = Decoder(pars['decoder'], pars['mode'])
        self.init_weights()

        self.att_layer = nn.Linear(**pars['att'])
        self.softmax = nn.Softmax(dim=-1)

        self.transitionCNN = nn.Conv2d(**pars['trans_layer'])
        self.mask = nn.Conv2d(**pars['trans_layer'])
        self.tanh = nn.Tanh()

    def forward(self, noisy, out_format='m2m'):
        bs, ch, fbins, nframes = noisy.size()
        compress, skips = self.encoder(noisy)
        denoised = self.decoder(compress, skips)
        if out_format == 'm2s_att':
            att_weights = self.softmax(self.att_layer(denoised.transpose(-1, -2)).transpose(-1, -2))
            denoised_frame = torch.sum(denoised * att_weights, dim=-1, keepdim=True)
            return denoised_frame
        elif out_format == 'm2s':
            denoised_frame = self.transitionCNN(denoised)
            return denoised_frame
        elif out_format == 'm2s_mask':
            # mask_frame = self.tanh(self.transitionCNN(denoised))
            mask_frame = self.transitionCNN(denoised)
            # print(mask_frame.shape)
            # print(noisy[:, :, :, nframes//2:nframes//2+1].shape)
            denoised_frame = mask_frame * noisy[:, :, :, nframes//2:nframes//2+1]
            return denoised_frame
        elif out_format == 'm2m':
            return denoised
        else:
            mask = self.transitionCNN(denoised)
            return self.tanh(mask)

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)


if __name__ == '__main__':
    u_net = UNet(pars)
    print(u_net)
    dummy_input = torch.rand((3, 1, 201, 11))
    # dummy_input = torch.rand((2, 1, , 65))
    output = u_net(dummy_input)
    print(output.shape)
    output = u_net(dummy_input, 'm2s_mask')
    print(output.shape)
