#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""
python3 local/predict_ema.py [model_dir] [input_wav_dir] [output_dir]
"""
import librosa
import numpy as np
import os
import s3prl.hub as hub
import soundfile as sf
import sys
import torch
import yaml

from scipy import stats
from tqdm import tqdm

from ats.bin.decode import ar_loop
from ats.utils import find_files, load_model, read_hdf5


hubert_device = 0
model_name = 'hubert_large_ll60k'
hubert_model = getattr(hub, model_name)() 
hubert_device = 'cuda:%d' % hubert_device
hubert_model = hubert_model.to(hubert_device)

def wav2mfcc(wav, sr, num_mfcc=13, n_mels=40, n_fft=320, hop_length=160):
    feat = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    feat = stats.zscore(feat, axis=None)
    return feat

exp_id = sys.argv[1]
if '_h2' in exp_id:
    input_modality = 'hubert'
else:
    input_modality = 'mfcc'
if exp_id.startswith('hprc'):
    interp_factor = 2
    hop_length = 160
else:
    interp_factor = 4
    hop_length = 80

inversion_model_d = 'exp/%s' % exp_id

device0 = 0

# Load Speech-to-EMA model
inversion_checkpoint_path = "%s/best_mel_ckpt.pkl" % inversion_model_d
inversion_config_path = "%s/config.yml" % inversion_model_d

# load config
with open(inversion_config_path) as f:
    inversion_config = yaml.load(f, Loader=yaml.Loader)

if torch.cuda.is_available():
    inversion_device = torch.device("cuda:%d" % device0)
else:
    inversion_device = torch.device("cpu")
inversion_model = load_model(inversion_checkpoint_path, inversion_config)
inversion_model.remove_weight_norm()
inversion_model = inversion_model.eval().to(inversion_device)

wav_d = sys.argv[2]
fs = os.listdir(wav_d)
fs = [f for f in fs if f.endswith('.wav')]

output_feats_d = sys.argv[3]
if not os.path.exists(output_feats_d):
    os.makedirs(output_feats_d)

with torch.no_grad():
    for f in tqdm(fs):
        p = os.path.join(wav_d, f)
        fid = f[:f.rfind('.')]
        output_art_path = os.path.join(output_feats_d, fid+'.npy')
        audio, sr = sf.read(p)
        if input_modality == 'hubert':
            wavs = [torch.from_numpy(audio).float().to(hubert_device)]
            states = hubert_model(wavs)["hidden_states"]
            feature = states[-1].squeeze(0)  # (seq_len, num_feats)
            target_length = len(feature)*interp_factor
            feature = torch.nn.functional.interpolate(feature.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False)
            feature = feature.transpose(1, 2).squeeze(0)  # (seq_len, num_feats)
            feat = feature.to(inversion_device)
            if "use_ar" in inversion_config["generator_params"] and inversion_config["generator_params"]["use_ar"]:
                pred = ar_loop(inversion_model, feat, inversion_config, normalize_before=False)
            else:
                pred = inversion_model.inference(feat, normalize_before=False)
        elif input_modality == 'mfcc':
            feat = wav2mfcc(audio, sr=sr, hop_length=hop_length).transpose()  # (seq_len, num_feats)
            feat = torch.tensor(feat, dtype=torch.float).to(inversion_device)
            if "use_ar" in inversion_config["generator_params"] and inversion_config["generator_params"]["use_ar"]:
                pred = ar_loop(inversion_model, feat, inversion_config, normalize_before=False)
            else:
                pred = inversion_model.inference(feat, normalize_before=False)
        np.save(output_art_path, pred.cpu().numpy())
