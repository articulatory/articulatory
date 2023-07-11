# -*- coding: utf-8 -*-

# Copyright 2023 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import os
import soundfile as sf
import s3prl.hub as hub
import torch

from tqdm import tqdm


def mk_hubert_features(wavd, hubertd, layer_idx=-1):
    """Extracts HUBERT features for the given waveforms.

    Each HUBERT feature has shape (sequence_length, 1024).
    The saved HUBERT features have a sampling rate 100 Hz.
    Assumes waveforms have a sampling rate of 16 kHz.

    Args:
        wavd: directory containing .wav files
        hubertd: directory to save hubert features
        layer_idx: index of HUBERT layer to extract features from
    """
    model_name = 'hubert_large_ll60k'
    model = getattr(hub, model_name)() 
    device = 'cpu'
    model=model.to(device)
    if not os.path.exists(hubertd):
        os.makedirs(hubertd)
    wavfs = os.listdir(wavd)
    wavfs = [f for f in wavfs if f.endswith('.wav')]
    for f in tqdm(wavfs):
        wavp = os.path.join(wavd, f)
        a, sr = sf.read(wavp)
        wavs = torch.from_numpy(a).float().to(device).unsqueeze(0)
        with torch.no_grad():
            states = model(wavs)["hidden_states"]
            feature = states[layer_idx].squeeze(0)
            target_length = len(feature)*2
            feature = torch.nn.functional.interpolate(feature.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False)
            feature = feature.transpose(1, 2).squeeze(0)
            oup = os.path.join(hubertd, f[:-4]+'.npy')
            np.save(oup, feature.cpu().numpy())
