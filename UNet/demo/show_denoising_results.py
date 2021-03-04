import matplotlib.pyplot as plt
import torch
import torchaudio
from matplotlib.widgets import Button
from os.path import join as pjoin
from scipy.io.wavfile import read as wavread
from torchaudio.functional import magphase
from playsound import playsound


class PlayAudio:
    def __init__(self, save2, feature_name, button_position, fromFeatures=False):
        self.audio_path = pjoin(save2, str(feature_name)) + '.wav' if not fromFeatures else pjoin(save2, str(feature_name)) + 'fromFeatures.wav'
        button_position = plt.axes(button_position)
        ICON_wav = plt.imread('./icon/listen_wav.png')
        ICON_spec = plt.imread('./icon/listen_spectrum.png')
        self.button = Button(button_position, '', image=ICON_wav) if not fromFeatures else Button(button_position, '', image=ICON_spec)

    def set_button(self):
        self.button.on_clicked(self._playaudio)

    def _playaudio(self, event):
        playsound(self.audio_path)


def show_results(wav_dir):
    denoised_path, noisy_path, clean_path, noisy_denoised_path = [pjoin(wav_dir, stream)
                                                                  for stream in
                                                                  ['denoised.wav', 'noisy.wav', 'clean.wav',
                                                                   'noisy_denoised.wav']]

    denoised_spec, noisy_spec, clean_spec, noisy_denoised_spec = \
        [torch.log(magphase(torch.stft(torchaudio.load(path, normalization=True)[0], win_length=400, hop_length=160, n_fft=400))[0] + 1e-5).squeeze()
         for path in
         [denoised_path, noisy_path, clean_path, noisy_denoised_path]]

    f_spec, t_spec = denoised_spec.shape

    fig = plt.figure()
    plt.subplot(411)
    plt.pcolormesh(range(t_spec), range(f_spec), noisy_spec, shading='auto')
    plt.title('Noisy')
    plt.subplot(412)
    plt.pcolormesh(range(t_spec), range(f_spec), clean_spec, shading='auto')
    plt.title('Clean')
    plt.subplot(413)
    plt.pcolormesh(range(t_spec), range(f_spec), denoised_spec, shading='auto')
    plt.title('Denoised')
    plt.subplot(414)
    plt.pcolormesh(range(t_spec), range(f_spec), noisy_denoised_spec, shading='auto')
    plt.title('Noisy => Denoised')

    playaudio_button_position_noisy = [0.8, 0.84, 0.1, 0.075]
    pa_noisy = PlayAudio(wav_dir, feature_name='noisy', button_position=playaudio_button_position_noisy)
    pa_noisy.set_button()

    playaudio_button_position_clean = [0.8, 0.6, 0.1, 0.075]
    pa_clean = PlayAudio(wav_dir, feature_name='clean', button_position=playaudio_button_position_clean)
    pa_clean.set_button()

    playaudio_button_position_denoised = [0.8, 0.35, 0.1, 0.075]
    pa_denoised = PlayAudio(wav_dir, feature_name='denoised', button_position=playaudio_button_position_denoised)
    pa_denoised.set_button()

    playaudio_button_position_noisy_denoised = [0.8, 0.11, 0.1, 0.075]
    pa_noisy_denoised = PlayAudio(wav_dir, feature_name='noisy_denoised', button_position=playaudio_button_position_noisy_denoised)
    pa_noisy_denoised.set_button()

    fig.tight_layout()

    plt.show()
