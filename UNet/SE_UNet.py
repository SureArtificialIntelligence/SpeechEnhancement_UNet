import torch
import torch.nn as nn
from torch import stft, istft
from UNet import UNet
from torchaudio.functional import magphase
from pars_UNet_light import pars_gh as pars


class SEUNet(nn.Module):
    def __init__(self, pars):
        super(SEUNet, self).__init__()
        self.unet = UNet(pars)

        self.fs = 16000
        self.win_len = 0.064
        self.hop_len = 0.016

    def forward(self, x):
        with torch.no_grad():
            x = self.format_in(x)
            spec = stft(x, int(self.fs*self.win_len), int(self.fs*self.hop_len))
            mag, pha = magphase(spec)
            mag = mag.unsqueeze(1)

        mask = self.unet(mag, 'mask')
        mag_masked = (mag * mask).squeeze()
        spec_processed = self.recover_spec(mag_masked, pha)
        denoised_wav = istft(spec_processed, int(self.fs*self.win_len), int(self.fs*self.hop_len))
        return denoised_wav

    def format_in(self, x):
        x = x.squeeze()
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return x

    def recover_spec(self, mag, pha):
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        spec_processed = torch.stack([real, imag], dim=-1)
        return spec_processed


if __name__ == '__main__':
    se_unet = SEUNet(pars)
    print(se_unet)

    dummy = torch.rand(1, 16384)
    output = se_unet(dummy)
    print(dummy.shape)
    print(output.shape)