import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import torchaudio
from torch import istft
from DataETL import DataETL
from torch.utils.data import DataLoader
from sklearn.feature_extraction.image import extract_patches_2d, extract_patches
from matplotlib.patches import Rectangle
from os.path import join as pjoin
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


"""
ToDo:
visulise['phases'] and phases should match
"""


class DataVisualisation:
    def __init__(self, dataset, save_audio2):
        self.existNoise = dataset.existNoise
        self.existPreserve = dataset.existPreserve
        self.existTarget = dataset.existTarget
        self.aux_stream = dataset.aux_stream
        self.context_mode = dataset.context_mode
        self.feature = dataset.feature
        self.fea_w_pha = dataset.fea_w_pha

        self.existSignalContext = dataset.existSignalContext
        self.existNoiseContext = dataset.existNoiseContext
        self.existPreserveContext = dataset.existPreserveContext if dataset.existPreserve else None

        self.win_size = dataset.win_size
        self.hop_size = dataset.hop_size
        self.slice_win = dataset.slice_win
        self.num_segments = dataset.num_segments

        data_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)
        iter_loader = iter(data_loader)
        data_batch, visualise = iter_loader.next()   # data_batch: features, phases, contexts, ctphases, targets, video

        self.features_segments = data_batch['features_segments']  # features_segments: mixed, signal, noise, target
        if self.fea_w_pha:
            self.phase_segments = data_batch['phases_segments']

        if self.context_mode is not None:
            self.contexts_trunk = data_batch['contexts']
            if self.fea_w_pha:
                self.ctphases_trunk = data_batch['ctphases']

        self.target = data_batch['target']
        self.features_segments['target'] = self.target

        if self.aux_stream:
            self.video = data_batch['video_segments']

        self.paths, self.audios, self.features, self.phases, self.contexts, self.ctphases, self.fs = visualise['paths'], visualise['audios'], visualise['features'], visualise['phases'], visualise['contexts'], visualise['ctphases'], visualise['fs']
        self.feature_all = self.features_segments.keys()  # mixed, signal, noise, preserve
        self.save_audio2 = save_audio2

    def visualise(self, feature_name=None, recover_audio=True, num_segments=1):
        if recover_audio:
            self._save_audio(self.audios, self.fs, self.feature_all, save2=self.save_audio2)
            if self.fea_w_pha:
                self._save_audio_recovered_from_features(self.features, self.phases, self.fs, self.feature_all, save2=self.save_audio2)
                if self.context_mode:
                    self._save_audio_recovered_from_features(self.contexts_trunk, self.ctphases_trunk, self.fs, self.contexts_trunk.keys(), save2=self.save_audio2)

        if feature_name is None:
            feature_names = self.feature_all
        else:
            feature_names = [feature_name]

        assert num_segments <= self.num_segments    # less segments to visualise than generated

        for feature_idx, feature_type in enumerate(feature_names):
            wav, feature, _ = self._contents(feature_type)
            sr = self.fs['audio_sr'].item()
            if self.feature != 'time-domain':
                F_f, T_f = feature.shape
            find_idxs = []
            for idx_segment in range(self.num_segments):
                _, _, segment = self._contents(feature_type, idx_segment)
                if self.aux_stream:
                    video = self._contents('video', idx_segment, idx_video_frame=0)  # each audio segment corresponds to several video frames.
                    video_fps = self.fs['video_fps'].item()
                    av_ratio = self.fs['av_ratio'].item()
                else:
                    video = plt.imread('./icon/tao.jpg')

                if self.feature == 'time-domain':
                    # print(segment.shape)
                    F_s, T_s = segment.shape
                    feature_patches = extract_patches(feature, patch_shape=T_s)
                else:
                    segment = segment.unsqueeze(-1) if len(segment.shape) == 1 else segment
                    F_s, T_s = segment.shape
                    assert F_f == F_s
                    feature_patches = extract_patches_2d(feature, (F_f, T_s))
                for idx_patch in range(len(feature_patches)):
                    if self.feature == 'time-domain':
                        find = feature_patches[idx_patch] == segment
                    else:
                        find = feature_patches[idx_patch] == segment
                    found = True if find.all() else False
                    if found:
                        find_idx = idx_patch
                        # print(find_idx)
                        find_idxs.append(find_idx)
                        break

            plt.figure(feature_idx)
            plt.subplot(221)
            plt.plot(wav)
            plt.title('{}, fs: {}'.format(feature_type, sr))

            plt.subplot(223)
            segment_colors = ['r', 'b', 'k']
            ax = plt.gca()
            if self.feature == 'time-domain':
                plt.plot(feature)
                for idx, find_idx in enumerate(find_idxs[:num_segments]):
                    # print(range(find_idx, find_idx + T_s))
                    # print(feature[find_idx:find_idx + T_s])
                    plt.plot(range(find_idx, find_idx + T_s), feature[find_idx:find_idx + T_s])

            else:
                plt.pcolormesh(range(T_f), range(F_f), feature, shading='auto')
                plt.ylim([0, F_f])
                for idx, find_idx in enumerate(find_idxs[:num_segments]):
                    rect = Rectangle((find_idx - 1, 0), T_s + 2, F_s, linewidth=2, edgecolor=segment_colors[idx],
                                     facecolor='none', alpha=0.9)
                    ax.add_patch(rect)

            plt.subplot(222)
            plt.imshow(video)
            if self.aux_stream:
                plt.title('fs: {}, av ratio: {}'.format(video_fps, av_ratio))
            else:
                # plt.title('No Image?')
                pass

            playaudio_button_position = [0.4, 0.8, 0.1, 0.075]
            playaudio_fromFeatures_button_position = [0.4, 0.38, 0.1, 0.075]

            if feature_type == 'signal':
                pa_signal = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_button_position)
                pa_signal.set_button()
                if self.feature != 'time-domain':
                    pa_signal_ff = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_fromFeatures_button_position, fromFeatures=True)
                    pa_signal_ff.set_button()
            elif feature_type == 'noise':
                pa_noise = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_button_position)
                pa_noise.set_button()
                if self.feature != 'time-domain':
                    pa_noise_ff = PlayAudio(self.save_audio2, feature_name=feature_type,
                                             button_position=playaudio_fromFeatures_button_position, fromFeatures=True)
                    pa_noise_ff.set_button()
            elif feature_type == 'preserve':
                pa_preserve = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_button_position)
                pa_preserve.set_button()
                if self.feature != 'time-domain':
                    pa_preserve_ff = PlayAudio(self.save_audio2, feature_name=feature_type,
                                             button_position=playaudio_fromFeatures_button_position, fromFeatures=True)
                    pa_preserve_ff.set_button()
            elif feature_type == 'mixed':
                pa_mixed = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_button_position)
                pa_mixed.set_button()
                if self.feature != 'time-domain':
                    pa_mixed_ff = PlayAudio(self.save_audio2, feature_name=feature_type,
                                             button_position=playaudio_fromFeatures_button_position, fromFeatures=True)
                    pa_mixed_ff.set_button()
            elif feature_type == 'target':
                pa_target = PlayAudio(self.save_audio2, feature_name=feature_type, button_position=playaudio_button_position)
                pa_target.set_button()
                if self.feature != 'time-domain':
                    pa_target_ff = PlayAudio(self.save_audio2, feature_name=feature_type,
                                             button_position=playaudio_fromFeatures_button_position, fromFeatures=True)
                    pa_target_ff.set_button()

            if self.context_mode and feature_type == 'mixed':
                if self.existSignalContext or self.context_mode=='auto':
                    context_signal = self.contexts_trunk['signal_context'].squeeze()
                    plt.subplot(2, 6, 10)
                    if self.feature == 'time-domain':
                        T_c = list(context_signal.shape)[0]
                        plt.plot(feature)
                        for idx, find_idx in enumerate(find_idxs[:num_segments]):
                            # T_c = T_c[0]
                            plt.plot(range(find_idx, find_idx + T_c), feature[find_idx:find_idx + T_c])
                    else:
                        F_c, T_c = context_signal.shape
                        plt.pcolormesh(range(T_c), range(F_c), context_signal, shading='auto')
                        playaudio_signal_context_button_position = [0.555, 0.38, 0.1, 0.075]
                    if self.feature != 'time-domain':
                        pa_signal_context_ff = PlayAudio(self.save_audio2, feature_name='signal_context',
                                                button_position=playaudio_signal_context_button_position, fromFeatures=True)
                        pa_signal_context_ff.set_button()

                if self.existNoiseContext or self.context_mode == 'auto':
                    context_noise = self.contexts_trunk['noise_context'].squeeze()
                    plt.subplot(2, 6, 11)
                    if self.feature == 'time-domain':
                        T_c = list(context_noise.shape)[0]
                        plt.plot(feature)
                        for idx, find_idx in enumerate(find_idxs[:num_segments]):
                            # T_c = T_c[0]
                            plt.plot(range(find_idx, find_idx + T_c), feature[find_idx:find_idx + T_c])
                    else:
                        F_c, T_c = context_noise.shape
                        plt.pcolormesh(range(T_c), range(F_c), context_noise, shading='auto')
                        playaudio_noise_context_button_position = [0.688, 0.38, 0.1, 0.075]
                    if self.feature != 'time-domain':
                        pa_noise_context_ff = PlayAudio(self.save_audio2, feature_name='noise_context',
                                                        button_position=playaudio_noise_context_button_position,
                                                        fromFeatures=True)
                        pa_noise_context_ff.set_button()

                if self.existPreserveContext or self.context_mode == 'auto':
                    context_preserve = self.contexts_trunk['preserve_context'].squeeze()
                    plt.subplot(2, 6, 12)
                    if self.feature == 'time-domain':
                        T_c = list(context_preserve.shape)[0]
                        plt.plot(feature)
                        for idx, find_idx in enumerate(find_idxs[:num_segments]):
                            # T_c = T_c[0]
                            plt.plot(range(find_idx, find_idx + T_c), feature[find_idx:find_idx + T_c])
                    else:
                        F_c, T_c = context_preserve.shape
                        plt.pcolormesh(range(T_c), range(F_c), context_preserve, shading='auto')
                        playaudio_preserve_context_button_position = [0.825, 0.38, 0.1, 0.075]
                    if self.feature != 'time-domain':
                        pa_preserve_context_ff = PlayAudio(self.save_audio2, feature_name='preserve_context',
                                                        button_position=playaudio_preserve_context_button_position,
                                                        fromFeatures=True)
                        pa_preserve_context_ff.set_button()

        plt.show()

    def _contents(self, feature_name, idx_segment=0, idx_video_frame=0):
        if feature_name == 'video':
            # shape of video, a list of tensors of frames, each tensor corresponds to an audio segment
            # print(self.video[idx_segment].shape)
            video = self.video[idx_segment][:, idx_video_frame].squeeze().numpy()
            return video
        else:
            wav = self.audios[feature_name].squeeze().numpy()
            feature = self.features[feature_name].squeeze().numpy()
            # print(feature_name)
            # print(len(self.features_segments[feature_name]))
            print(idx_segment)
            segment = self.features_segments[feature_name][idx_segment].squeeze(0).numpy()
            return wav, feature, segment

    def _save_audio(self, audios, fs, feature_names, save2):
        for feature_name in feature_names:
            audio = audios[feature_name]
            audio = audio.squeeze(0) if len(audio.shape) > 2 else audio
            torchaudio.save(pjoin(save2, feature_name) + '.wav', audio, fs['audio_sr'].item())

    def _save_audio_recovered_from_features(self, features, phases, fs, feature_names, save2):
        for feature_name in feature_names:
            feature = features[feature_name]
            if self.feature.startswith('log-'):
                feature = torch.exp(feature)
            if self.feature.endswith('spectrogram'):
                feature = torch.sqrt(feature)
            # padd_len = int((self.slice_win - 1) / 2)
            # feature = feature[:, :, padd_len:-padd_len]
            phase = phases[feature_name]
            # phase = phase[:, :, padd_len:-padd_len]
            feature_real = (feature * torch.cos(phase)).unsqueeze(3)
            feature_imag = (feature * torch.sin(phase)).unsqueeze(3)
            feature = torch.cat([feature_real, feature_imag], dim=-1)
            nfft = int(self.win_size*fs['audio_sr'].item())
            waveform = istft(feature, nfft, win_length=int(self.win_size*fs['audio_sr'].item()), hop_length=int(self.hop_size*fs['audio_sr'].item()))
            torchaudio.save(pjoin(save2, str(feature_name)) + 'fromFeatures.wav', waveform, fs['audio_sr'].item())


if __name__ == '__main__':
    signal_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/test.pkl'
    noise_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/test.pkl'
    # data_etl = DataETL('train', signal_filelist_path=signal_seeds, noise_filelist_path=noise_seeds,
    #                    feature='log-magnitude', short_version=1, slice_win=5, target_scope=2, num_segments=5,
    #                    mute_random=True, one_sample=True, visualise=True)
    data_etl = DataETL('test', signal_filelist_path=signal_seeds, noise_filelist_path=noise_seeds,
                       feature='time-domain', short_version=2, slice_win=16384,
                       mute_random=True, mute_random_snr=True, padding_slice=False, visualise=True)

    audio_save2 = './audio_demo/'
    dv = DataVisualisation(data_etl, audio_save2)
    dv.visualise(num_segments=3)
    print('--------EOF---------')
