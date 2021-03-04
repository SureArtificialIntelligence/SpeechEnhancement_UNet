from __future__ import division
import random
import math
import pickle as p
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.functional import magphase
from torchvision.io import read_video as vidread

"""
ToDo:
1. split magnitude and phase
2. record in noise and preserve
"""


class DataETL(Dataset):
    def __init__(self, partition, signal_filelist_path, noise_filelist_path=None, preserve_filelist_path=None,
                 target_filelist_path=None, visualise=False, short_version=False,
                 mute_random=False, mute_random_snr=False, one_sample=False,
                 **kwargs):
        """

        :param partition: 'train' / 'valid' / 'test'
        :param signal_filelist_path:
        :param noise_filelist_path:
        :param preserve_filelist_path:
        :param target_filelist_path:

        case1: signal & target ------------------> denoising: noisy = signal; target = target
        case2: signal & noise -------------------> denoising: noisy = signal + noise; target = signal
        case3: signal & noise & preserve --------> selective noise suppression: mixture = signal + noise + preserve;
                                                                                negative noise = noise;
                                                                                positive noise = preserve;
                                                                                target = signal + preserve

        :param short_version: # utterance
        :param mute_random: control the mixture way of signal and noise, etc
        :param one_sample: # True, truncate the same segments.

        """

        super(DataETL).__init__()

        self.feature = kwargs['feature'] if 'feature' in kwargs.keys() else 'time-domain'
        self.win_size = kwargs['win_size'] if 'win_size' in kwargs.keys() else 0.025
        self.hop_size = kwargs['hop_size'] if 'hop_size' in kwargs.keys() else 0.010
        self.slice_win = kwargs['slice_win'] if 'slice_win' in kwargs.keys() else 35
        self.num_segments = kwargs['num_segments'] if 'num_segments' in kwargs.keys() else 5
        self.target_mode = kwargs['target_mode'] if 'target_mode' in kwargs.keys() else 'centre'
        self.target_scope = kwargs['target_scope'] if 'target_scope' in kwargs.keys() else 0
        self.padding_slice = kwargs[
            'padding_slice'] if 'padding_slice' in kwargs.keys() else True  # apply only to frequency domain features
        self.context_mode = kwargs[
            'context_mode'] if 'context_mode' in kwargs.keys() else None  # 'auto' / 'man' / None
        self.context_win = kwargs['context_win'] if 'context_win' in kwargs.keys() else 100
        self.visualise = kwargs['visualise'] if 'visualise' in kwargs.keys() else False

        self.given_snr_dict = kwargs['given_snr_dict'] if 'given_snr_dict' in kwargs.keys() else None

        # Partition indicates the dataset belongings: 'train' => training / 'valid' => validation / 'test' => test
        self.partition = partition
        self.existNoise = True if noise_filelist_path else False
        self.existPreserve = True if preserve_filelist_path else False
        self.existTarget = True if target_filelist_path else False
        # self.existRecord = True if self.context_mode == 'man' else False

        # Load filelists from their paths
        # Seed in signal_filelist can contain different streams:
        #
        # Signal
        # idx    content
        #   0 => audio stream
        #   1 => audio context
        #   2 => video
        #
        # Noise
        # idx    content
        #   0 => audio stream
        #   1 => audio context
        #
        # Preserve
        # idx    content
        #   0 => audio stream
        #   1 => audio context
        #   2 => video

        self.signal_filelist = self.load_filelist(signal_filelist_path)
        self.num_speech = len(self.signal_filelist)

        if self.existNoise:
            self.noise_filelist = self.load_filelist(noise_filelist_path)
            self.num_noise = len(self.noise_filelist)
        if self.existPreserve:
            self.preserve_filelist = self.load_filelist(preserve_filelist_path)
            self.num_preserve = len(self.preserve_filelist)
        if self.existTarget:
            self.target_filelist = self.load_filelist(target_filelist_path)
            self.num_target = len(self.target_filelist)

        self.aux_stream = True if len(self.signal_filelist[0]) > 1 and (
                    self.signal_filelist[0][0][-4:] != self.signal_filelist[0][-1][-4:]) else False
        self.existSignalContext = True if len(self.signal_filelist[0]) > 1 and (
                    self.signal_filelist[0][0][-4:] == self.signal_filelist[0][1][-4:]) else False
        self.existNoiseContext = True if len(self.noise_filelist[0]) > 1 and (
                    self.noise_filelist[0][0][-4:] == self.noise_filelist[0][1][-4:]) else False
        if self.existPreserve:
            self.existPreserveContext = True if len(self.preserve_filelist[0]) > 1 and (
                        self.preserve_filelist[0][0][-4:] == self.preserve_filelist[0][1][-4:]) else False
        self.existContext = self.existSignalContext or self.existNoiseContext or self.existPreserveContext if self.existPreserve else self.existSignalContext or self.existNoiseContext
        assert self.existContext == (self.context_mode == 'man')
        self.fea_w_pha = not (self.feature == 'time-domain' or self.feature == 'mfcc')

        self.visualise = visualise
        self.short_version = short_version
        self.mute_random = mute_random
        self.mute_random_snr = mute_random_snr
        self.one_sample = one_sample

    def __getitem__(self, idx):
        visualisation_sr = {}
        visualisation_paths = {}
        visualisation_audios = {}
        visualisation_features = {}
        visualisation_phases = {}
        visualisation_contexts = {}
        visualisation_ctphases = {}
        visualisation_info = {}

        if self.aux_stream:
            video_path = self.signal_filelist[idx][-1]
            visualisation_paths['video'] = video_path

        audio_paths = {}

        # signal
        signal_paths, audio_paths, visualisation_paths = self.parser('signal', self.signal_filelist,
                                                                     self.partition, idx, audio_paths,
                                                                     visualisation_paths)

        # noise
        if self.existNoise:
            noise_paths, audio_paths, visualisation_paths = self.parser('noise', self.noise_filelist,
                                                                        self.partition, idx, audio_paths,
                                                                        visualisation_paths)

        # preserve
        if self.existPreserve:
            preserve_paths, audio_paths, visualisation_paths = self.parser('preserve', self.preserve_filelist,
                                                                           self.partition, idx, audio_paths,
                                                                           visualisation_paths)

        if self.existTarget:
            target_path = self.target_filelist[idx]
            visualisation_paths['target'] = target_path
            audio_paths['target'] = target_path

        #################################  manually modification  ###################################
        # load in streams
        audio_wavs, wav_sr = self.reader(audio_paths)
        audio_wavs = self.length_channel_matcher(audio_wavs)
        # create mixture
        # if there's no target reference, we need to mix signal and noise to create mixture
        snr = self.snr_generator(path_name=audio_paths['signal'] + audio_paths['noise'], given_snr_dict=self.given_snr_dict)
        snrs = {'sn': snr}
        if self.existPreserve:
            snr_preserve = self.snr_generator(path_name=audio_paths['signal'] + audio_paths['preserve'], given_snr_dict=self.given_snr_dict)
            snrs['sp'] = snr_preserve
        audio_wavs, Ks = self.mixer(audio_wavs,
                                    snrs)  # mixed, signal, noise, preserve, signal_context, noise_context, preserve_context
        audio_wavs = self.add_target(audio_wavs)
        features = self.stream_processor(audio_wavs, feature=self.feature, sr=wav_sr, win_size=self.win_size,
                                         hop_size=self.hop_size)
        if self.fea_w_pha:
            features, phases = features[0], features[1]

        if self.context_mode:
            if self.fea_w_pha:
                (features, phases), (contexts, contexts_phases) = self.context(features, phases)
                if self.visualise:
                    visualisation_ctphases = contexts_phases
                    visualisation_phases = phases
            else:
                features, contexts = self.context(features)

            if self.visualise:
                visualisation_contexts = contexts

        if self.padding_slice:
            features = self.padding(features, slice_win=self.slice_win)
            if self.fea_w_pha:
                phases = self.padding(phases, slice_win=self.slice_win)

        if self.visualise:
            visualisation_audios = audio_wavs
            visualisation_features = features
            if self.fea_w_pha:
                visualisation_phases = phases

        if self.fea_w_pha:
            features_segments, phases_segments, idxes_segments = self.feature_segmentation(features, phases,
                                                                                           slice_win=self.slice_win,
                                                                                           num_segments=self.num_segments)
        else:
            features_segments, idxes_segments = self.feature_segmentation(features, slice_win=self.slice_win,
                                                                          num_segments=self.num_segments)

        target_segments = features_segments['target']
        target_frames = self.slice_target(target_segments, slice_win=self.slice_win, mode=self.target_mode,
                                          scope=self.target_scope)

        if self.context_mode is not None:
            if self.fea_w_pha:
                A = {'features_segments': features_segments,
                     'phases_segments': phases_segments,
                     'contexts': contexts,
                     'ctphases': contexts_phases,
                     'target': target_frames}
            else:
                A = {'features_segments': features_segments,
                     'contexts': contexts,
                     'target': target_frames}

        else:
            if self.fea_w_pha:
                A = {'features_segments': features_segments,
                     'phases_segments': phases_segments,
                     'target': target_frames}
            else:
                A = {'features_segments': features_segments,
                     'target': target_frames}

        if self.visualise:
            visualisation_sr['audio_sr'] = wav_sr

        if self.aux_stream:
            video_frames, video_fps = self.video_reader(video=video_path)
            # video_frames = self.video_processor(video_frames)
            # video_frames = self.video_advanced_processor(video_frames)
            if self.feature == 'time-domain':
                av_ratio = wav_sr / video_fps
            else:
                av_ratio = wav_sr / video_fps * self.hop_size  # how many frames match to one video image
            video_segments = self.video_segmentation(video_frames, audio_slice_win=self.slice_win, av_ratio=av_ratio,
                                                     start_idxes=idxes_segments)
            V = {'video_segments': video_segments}
            if self.visualise:
                visualisation_sr['video_fps'] = video_fps
                visualisation_sr['av_ratio'] = av_ratio
        else:
            V = {}

        if self.visualise:
            for k, v in zip(['paths', 'audios', 'features', 'phases', 'contexts', 'ctphases', 'fs'],
                            [visualisation_paths, visualisation_audios, visualisation_features, visualisation_phases,
                             visualisation_contexts, visualisation_ctphases, visualisation_sr]):
                visualisation_info[k] = v
            return self.AV_aggregator(A, V), visualisation_info
        else:
            return self.AV_aggregator(A, V)

        ##############################################################################################

    def parser(self, stream, filelist, partition, idx, audio_paths, visualisation_paths):
        num_files = len(filelist)
        if stream != 'signal':
            if partition == 'train' and not self.mute_random:
                idx = random.randint(0, num_files - 1)
            else:
                idx = idx
        # else:
        #    if self.aux_stream:
        #        filelist[idx] = filelist[idx][:-1]
        # print(filelist[idx])
        path = filelist[idx][0]
        audio_paths[stream] = path
        visualisation_paths[stream] = path
        paths = [path]
        if len(filelist[idx]) > 1 and not self.aux_stream:
            context_path = filelist[idx][1]
            audio_paths[stream + '_context'] = context_path
            visualisation_paths[stream + 'context'] = context_path
            paths.append(context_path)

        return paths, audio_paths, visualisation_paths

    def reader(self, audio_paths):
        """

        :param signal: clean signal path
        :param noise:  noise path
        :param video: video path
        :param preserve:
        :return:
        """

        audio_waveform = {}
        for k, v in audio_paths.items():
            audio_waveform[k], audio_sr = torchaudio.load(v, normalization=True)
            if k == 'signal':
                sr = audio_sr
            else:
                assert sr == audio_sr

        return audio_waveform, sr

    # target
    def add_target(self, audio_wavs):
        """

        :param signal:
        :param noise:
        :param preserve:
        :return:
        """

        if self.existTarget:
            target = audio_wavs['target']
        elif self.existPreserve:
            target = audio_wavs['signal'] + audio_wavs['preserve']
        else:
            target = audio_wavs['signal']
        target = target.squeeze()
        audio_wavs['target'] = target
        return audio_wavs

    # mixer
    def length_channel_matcher(self, audio_wavs):

        def match2signal(signal, stream):
            _, signal_len = signal.shape
            audio_wav = audio_wavs[stream]
            channels, wav_len = audio_wav.shape
            if channels > 1:
                audio_wav = torch.mean(audio_wav, dim=0)
            if signal_len < wav_len:
                audio_wav = audio_wav[:, :signal_len]
            else:
                while wav_len < signal_len:
                    audio_wav = torch.cat((audio_wav, audio_wav[:, :signal_len - wav_len]), dim=1)
                    wav_len = audio_wav.shape[1]

            return audio_wav

        signal = audio_wavs['signal']
        signal_channels, signal_len = signal.shape

        # audio_wavs['signal']
        if signal_channels > 1:
            signal = torch.mean(signal, dim=0)
        noise = match2signal(signal, 'noise')
        audio_wavs['noise'] = noise

        if self.existPreserve:
            preserve = match2signal(signal, 'preserve')
            audio_wavs['preserve'] = preserve

        for k, v in audio_wavs.items():
            audio_wavs[k] = audio_wavs[k].squeeze()
        return audio_wavs

    def mixer(self, audio_wavs, snrs):
        """

        :param signal:
        :param noise:
        :param snr:
        :param preserve:
        :param snr_preserve:
        :return:
        """
        # compute the average energy for each sample
        signal = audio_wavs['signal']
        noise = audio_wavs['noise']
        signal_len = len(signal)
        noise_len = len(noise)
        signal_E = torch.sum(signal ** 2) / signal_len
        noise_E = torch.sum(noise ** 2) / noise_len

        snr = snrs['sn']
        SNR = 10 ** (snr / 10)
        K_sn = torch.sqrt(SNR / (
                    signal_E / noise_E))  # n/K_sn = n / (snr / (s/n)) = n / (snr * n / s) = n*s/(snr*n) = s/snr => s/(s/snr) = snr
        noise = noise / K_sn
        mixed = signal + noise
        audio_wavs['noise'] = noise
        if self.existNoiseContext:
            audio_wavs['noise_context'] = audio_wavs['noise_context'] / K_sn
        Ks = {'sn': K_sn}

        if self.existPreserve:
            preserve = audio_wavs['preserve']
            preserve_len = len(preserve)
            preserve_E = torch.sum(preserve ** 2) / preserve_len
            snr_preserve = snrs['sp']
            SNR_preserve = 10 ** (snr_preserve / 10)
            K_sp = torch.sqrt(SNR_preserve / (signal_E / preserve_E))
            preserve = preserve / K_sp
            mixed += preserve
            audio_wavs['preserve'] = preserve
            if self.existPreserveContext:
                audio_wavs['preserve_context'] = audio_wavs['preserve_context'] / K_sn
            Ks['sp'] = K_sp
        audio_wavs['mixed'] = mixed

        return audio_wavs, Ks

    def stream_processor(self, audio_wavs, feature='time-domain', sr=16000, win_size=None, hop_size=None):
        """

        :param signals: a list of signals, i.e, [signal, noise, mixed, target, preserve]
        :param feature: features to be extracted for the above signals, such as raw, stft => spectrogram, phase, mfcc
        :return:
        """

        # general parameters for stft-related extracting
        hop_length = int(sr * hop_size)
        win_length = int(sr * win_size)
        n_fft = int(sr * win_size)
        n_mels = 128
        n_mfcc = 40

        def feature_converter(feature):
            if feature.endswith('spectrogram'):
                converter = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                              hop_length=hop_length, power=2)
            elif feature.endswith('melspectogram'):
                converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                                                                 win_length=win_length, hop_length=hop_length,
                                                                 n_mels=n_mels)
            elif feature == 'melscale':
                converter = torchaudio.transforms.MelScale(sample_rate=sr, n_mels=n_mels)
            elif feature == 'mfcc':
                converter = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc)
            else:
                converter = None
                print('Wrong feature type!')
            return converter

        feature_from_timedomain = ['time-domain']
        feature_from_freqdomain = ['magnitude', 'log-magnitude', 'spectrogram', 'log-spectrogram', 'melspectrogram',
                                   'log-melspectrogram', 'mfcc', 'melscale']
        assert feature in feature_from_timedomain or feature in feature_from_freqdomain
        if feature == 'time-domain':
            features = audio_wavs

        else:
            streams_stft = {}
            phases = {}
            features = {}
            for k, v in audio_wavs.items():
                streams_stft[k] = torch.stft(audio_wavs[k], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                phases[k] = magphase(streams_stft[k])[1]
                if feature.endswith('magnitude'):
                    features[k] = magphase(streams_stft[k])[0]
                else:
                    f_convert = feature_converter(feature)
                    features[k] = f_convert(audio_wavs[k])
                if feature.startswith('log-'):
                    features[k] = torch.log(features[k] + 1e-5)

            if self.feature != 'mfcc':
                features = [features, phases]

        return features

    def padding(self, features, slice_win):
        features_padd = {}
        padd_len = int((slice_win - 1) / 2)
        for k, v in features.items():
            F, T = v.shape
            feature_padd = torch.zeros((F, T + slice_win - 1))
            feature_padd[:, padd_len:T + padd_len] = v
            features_padd[k] = feature_padd

        # print(features_padd[0][1][:, 17:-17] == features[1])
        return features_padd

    def context(self, features, phases=None):
        features_update = {}
        phases_update = {}
        contexts = {}
        contexts_phases = {}

        if self.feature == 'time-domain':
            for k, v in features.items():
                features[k] = v.unsqueeze(dim=0)

        if self.existContext:
            for k in features.keys():
                if k.endswith('_context'):
                    contexts[k] = features[k][:, :self.context_win]
                    if phases is not None:
                        contexts_phases[k] = phases[k][:, :self.context_win]
                else:
                    features_update[k] = features[k]
                    if phases is not None:
                        phases_update[k] = phases[k]

        elif self.context_mode == 'auto':
            for k in features.keys():
                feature = features[k]
                if phases is not None:
                    phase = phases[k]
                if self.partition == 'train':
                    len = feature.shape[1]
                    ss = random.randint(0, len - self.context_win)
                    to = ss + self.context_win
                    features_update[k] = torch.cat([feature[:, :ss], feature[:, to:]], dim=1)
                    if phases is not None:
                        phases_update[k] = torch.cat([phase[:, :ss], phase[:, to:]], dim=1)
                    contexts[k + '_context'] = feature[:, ss:to]
                    if phases is not None:
                        contexts_phases[k + '_context'] = phase[:, ss:to]
                else:
                    contexts[k + '_context'], features_update[k] = feature[:, :self.context_win], feature[:,
                                                                                                  self.context_win:]
                    if phases is not None:
                        contexts_phases[k + '_context'], phases_update[k] = phase[:, :self.context_win], phase[:,
                                                                                                         self.context_win:]

        if phases is None:
            return features_update, contexts
        else:
            return (features_update, phases_update), (contexts, contexts_phases)

    def feature_segmentation(self, features, phases=None, slice_win=None, num_segments=None):
        """

        :return: input segment, audio context
        """
        # if self.feature == 'time-domain':
        #     slice_dim = 0
        # else:
        slice_dim = 1

        features_name, features_update = [], []
        for k, v in features.items():
            features_name.append(k)
            features_update.append(v)
        features = features_update
        win_frames_shift = 1
        if self.feature == 'time-domain':
            if len(features[0].shape) == 1:
                features = [feature.unsqueeze(0) for feature in features]
            win_frames_shift = slice_win

        features_segments = [feature.unfold(slice_dim, slice_win, win_frames_shift) for feature in features]  # dimension, size, step
        num_streams = len(features_segments)  # signal_segments, noise_segments,......

        if phases is not None:
            phases_name, phases_update = [], []
            for k, v in phases.items():
                phases_name.append(k)
                phases_update.append(v)
            phases = phases_update
            phases_segments = [phase.unfold(slice_dim, slice_win, 1) for phase in phases]
            num_phases = len(phases_segments)
            [features_segments.append(phase_segments) for phase_segments in phases_segments]

        total_segments = features_segments[0].shape[slice_dim]
        if self.partition == 'train':
            idx_segments = random.sample(range(total_segments), num_segments)
            # if self.mute_random and not self.one_sample:
            #     idx_segments = range(18, 23)
            if self.mute_random:
                if self.one_sample:
                    idx_segments = range(int((self.slice_win - 1) / 2) + 1,
                                         int((self.slice_win - 1) / 2) + 1 + num_segments)  # first segment wo front padding
                else:
                    idx_segments = random.sample(range(total_segments), num_segments)

        else:
            idx_segments = range(total_segments)
            num_segments = total_segments

        features_segments = [feature_segments[:, idx_segment, :] for feature_segments in features_segments for
                             idx_segment in idx_segments]
        features_segments = [features_segments[i:i + num_segments] for i in range(0, len(features_segments),
                                                                                  num_segments)]  # stream x num_segments x ( F x T ), F=1 when time-domain

        if phases is not None:
            phases_segments = features_segments[-num_phases:]
            features_segments = features_segments[:-num_phases]

        assert len(features_segments) == num_streams

        features_segments_update = {}
        phases_segments_update = {}
        for i in range(len(features_segments)):
            features_segments_update[features_name[i]] = features_segments[i]
            if phases is not None:
                phases_segments_update[features_name[i]] = phases_segments[i]

        if phases is not None:
            return features_segments_update, phases_segments_update, idx_segments
        else:
            return features_segments_update, idx_segments

    def snr_generator(self, snr_range=[0, 5, 10, 15, 20], path_name=None, given_snr_dict=None):
        if given_snr_dict is not None:
            data_name = path_name.split('/')[-1][:-4]
            snr = given_snr_dict[data_name]
        else:
            if self.partition == 'train':
                snr_idx = random.randint(0, len(snr_range) - 1)
                if self.mute_random_snr:
                    snr_idx = 1
            else:  #self.partition == 'valid' or self.partition == 'test':
                snr_idx = int(hashlib.md5(path_name.encode('utf-8')).hexdigest()[:4], 16) % len(snr_range)
            snr = snr_range[snr_idx]
        return snr

    def slice_target(self, target_segments, slice_win, mode='centre', scope=0):
        if mode == 'centre':
            centre = (slice_win - 1) // 2
        elif mode == 'last':
            centre = slice_win - 1
            scope = 0
        else:
            centre = None
            print('Please check centre of target!')

        target_frames = [target_segment[:, centre - scope:centre + scope + 1] if len(target_segment.shape) == 2
                         else target_segment[centre - scope:centre + scope + 1] for target_segment in target_segments]

        return target_frames

    """
    video stream
    """

    def video_reader(self, video):
        # video stream
        if self.aux_stream:
            video_frames, _, meta_info = vidread(video)
            video_fps = int(meta_info['video_fps'])
        else:
            video_frames, video_fps = [], []
        return video_frames, video_fps

    def video_processor(self, signals, feature='au'):
        """

        :param signals: video stream
        :param feature: type of feature, such as 'au', 'landmark',....
        :return: features
        """
        return signals

    def video_advanced_processor(self, signals):
        """

        :param partition: random/consecutive crop according to sequence index
        :return: advanced features
        """
        return signals

    def video_segmentation(self, video_frames, audio_slice_win, av_ratio, start_idxes):
        vid_starts = [math.floor(start_idx / av_ratio) for start_idx in start_idxes]
        vid_frames_per_segment = int(audio_slice_win / av_ratio)
        video_segments = [video_frames[vid_start:vid_start + vid_frames_per_segment + 1] for vid_start in vid_starts]
        return video_segments

    """
    audio/video aggregation
    """

    def AV_aggregator(self, A, V):
        """

        :param A: audio features
        :param V: video features
        :return: av features, target, visualisationAVdata
        """
        aggregate_feaures = {}
        for k, v in A.items():
            aggregate_feaures[k] = v
        for k, v in V.items():
            aggregate_feaures[k] = v

        return aggregate_feaures

    def __len__(self):
        if self.short_version:
            return self.short_version
        return len(self.signal_filelist)

    def load_filelist(self, filelist_pkl):
        with open(filelist_pkl, 'rb') as pkl_file:
            filelist = p.load(pkl_file)
        return filelist


if __name__ == '__main__':
    signal_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Speech/pkls/train.pkl'
    noise_seeds = '/nas/staff/data_work/Sure/Edinburg_Noisy_Speech_Database/Noise/pkls/train.pkl'

    ################################## check if features have phases infomation ########################################
    feature_type_list = ['time-domain', 'magnitude', 'log-magnitude', 'spectrogram', 'log-spectrogram',
                         'melspectrogram',
                         'mfcc', 'melscale']
    feature_type_wo_phases = ['magnitude', 'log-magnitude', 'spectrogram', 'log-spectrogram', 'melspectrogram']
    feature_type = 'time-domain'
    # feature_type = 'log-magnitude'
    fea_w_pha = feature_type in feature_type_wo_phases
    ####################################################################################################################

    data_etl = DataETL('train', signal_filelist_path=signal_seeds, noise_filelist_path=noise_seeds,
                       preserve_filelist_path=noise_seeds, feature=feature_type, num_segments=10, context_mode='auto',
                       visualise=False, short_version=4)
    data_loader = DataLoader(data_etl, shuffle=True, batch_size=2, num_workers=4)
    for idx_batch, batch in enumerate(data_loader):
        print('Elements in one batch: {}'.format(list(batch.keys())))
        features_segments_ = batch['features_segments']
        contexts_ = batch['contexts']
        target_ = batch['target']
        if 'video_segement' in batch.keys():
            video_ = batch['video_segments']

        if fea_w_pha:
            phase_segments_ = batch['phases_segments']
            ctphases_ = batch['ctphases']

        mixed_ = features_segments_['mixed']
        signal_ = features_segments_['signal']
        noise_ = features_segments_['noise']
        preserve_ = features_segments_['preserve']
        # target_ = features_segments_['target']

        print('--------------feature map input--------------')
        segment_num = 0
        print(mixed_[segment_num].shape)
        print(signal_[segment_num].shape)
        print(noise_[segment_num].shape)
        print(preserve_[segment_num].shape)
        print('---------------context input------------')
        signal_context_ = contexts_['signal_context']
        noise_context_ = contexts_['noise_context']
        preserve_context_ = contexts_['preserve_context']

        print(signal_context_.shape)
        print(noise_context_.shape)
        print(preserve_context_.shape)

    print('--------EOF---------')
