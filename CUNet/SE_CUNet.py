import torch
import torch.nn as nn
from torch import stft, istft
from CUNet import ComplexUNet
from torchaudio.functional import magphase
from pars_complexUNet_light import pars_gh as pars


class SECUNet(nn.Module):
    def __init__(self, pars):
        super(SECUNet, self).__init__()
        self.cunet = ComplexUNet(pars)

        self.fs = 16000
        self.win_len = 0.064
        self.hop_len = 0.016

    def forward(self, x):
        with torch.no_grad():
            x = self.format_in(x)
            spec = stft(x, int(self.fs * self.win_len), int(self.fs * self.hop_len))
            spec_real, spec_imag = spec[:, :, :, 0], spec[:, :, :, 1]
            spec_input = torch.stack([spec_real, spec_imag], dim=1).unsqueeze(2)

        mask = self.cunet(spec_input)
        # print(mask.shape)
        spec_masked = (spec_input * mask).squeeze(dim=2)
        # print(spec_masked.shape)
        spec_real_processed = spec_masked[:, 0, :, :]
        spec_imag_processed = spec_masked[:, 1, :, :]
        spec_processed = torch.stack([spec_real_processed, spec_imag_processed], dim=-1)
        denoised_wav = istft(spec_processed, int(self.fs * self.win_len), int(self.fs * self.hop_len))
        return denoised_wav

    def format_in(self, x):
        x = x.squeeze()
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return x


if __name__ == '__main__':
    se_cunet = SECUNet(pars)
    print(se_cunet)

    dummy = torch.rand(3, 16384)
    output = se_cunet(dummy)
    print(dummy.shape)
    print(output.shape)