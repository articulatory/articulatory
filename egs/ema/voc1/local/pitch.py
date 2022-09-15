#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Extracts pitch and periodicity information.

E.g., `python3 local/pitch.py downloads/emadata/cin_us_mngu0 --hop 80`
        outputs pitch data in `downloads/emadata/cin_us_mngu0/pitch`

Based on Max Morrison's https://github.com/descriptinc/cargan/blob/master/cargan/preprocess/pitch.py.
"""

import argparse
import functools
import numpy as np
import os
import sys
import soundfile as sf
import torch
import torchaudio
import torchcrepe

from tqdm import tqdm


def from_audio(audio, sample_rate=22050, gpu=None, hopsize=256, num_fft=1024, fmin=50, fmax=550):
    """Preprocess pitch from audio"""
    # Target number of frames
    target_length = audio.shape[1] // hopsize
    
    # Resample
    if sample_rate != torchcrepe.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate,
                                                   torchcrepe.SAMPLE_RATE)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
    
    # Resample hopsize
    hopsize = int(hopsize * (torchcrepe.SAMPLE_RATE / sample_rate))

    # Pad
    padding = int((num_fft - hopsize) // 2)
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect').squeeze(0)

    # Estimate pitch
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        fmin=fmin,
        fmax=fmax,
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device='cpu' if gpu is None else f'cuda:{gpu}',
        pad=False)

    # Set low energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False)

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('d')
    parser.add_argument('--hop', type=int, default=110)
    args = parser.parse_args()

    gpu = 1 # was None

    if not os.path.exists(args.d):
        downloads_subdir = os.path.join('downloads', args.d)
    else:
        downloads_subdir = args.d
    wav_dir = os.path.join(downloads_subdir, 'wav')
    fs = os.listdir(wav_dir)
    fs = [f for f in fs if f.endswith('.wav')]
    fs = sorted(fs)
    pitch_dir = os.path.join(downloads_subdir, 'pitch')
    periodicity_dir = os.path.join(downloads_subdir, 'periodicity')
    if not os.path.exists(pitch_dir):
        os.makedirs(pitch_dir)
    if not os.path.exists(periodicity_dir):
        os.makedirs(periodicity_dir)
    device = torch.device("cuda:1")
    torchcrepe.load.model(device)
    min_pitch = 1e6
    max_pitch = -1e6
    min_periodicity = 1e6
    max_periodicity = -1e6
    for f in tqdm(fs):
        wav_p = os.path.join(wav_dir, f)
        a, sr = torchaudio.load(wav_p)
        pitch, periodicity = from_audio(a, sample_rate=sr, gpu=gpu, hopsize=args.hop, num_fft=1024, fmin=50, fmax=550)
        pitch = pitch[0].cpu().numpy()
        periodicity = periodicity[0].cpu().numpy()
        min_pitch = min(min_pitch, np.min(pitch))
        max_pitch = max(min_pitch, np.max(pitch))
        min_periodicity = min(min_periodicity, np.min(periodicity))
        max_periodicity = max(min_periodicity, np.max(periodicity))
        pitch_p = os.path.join(pitch_dir, f.replace('.wav', '.npy'))
        periodicity_p = os.path.join(periodicity_dir, f.replace('.wav', '.npy'))
        np.save(pitch_p, pitch)
        np.save(periodicity_p, periodicity)
    with open(os.path.join(downloads_subdir, 'pitch_minmax.txt'), 'w+') as ouf:
        ouf.write('%f %f\n' % (min_pitch, max_pitch))
    with open(os.path.join(downloads_subdir, 'periodicity_minmax.txt'), 'w+') as ouf:
        ouf.write('%f %f\n' % (min_periodicity, max_periodicity))
