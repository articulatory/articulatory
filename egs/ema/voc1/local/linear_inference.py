#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""Speech to EMA with linear regression model.

Assumes waveform is 16000 Hz.

python3 linear_inference.py \
    /path/to/waveform/file \
    /path/to/linear.joblib \
    /path/to/predicted/ema
"""

import numpy as np
import os
import soundfile as sf
import sys
import s3prl.hub as hub
import torch

from joblib import load


model_name = 'wavlm_large'
model = getattr(hub, model_name)() 
device = 'cuda:0'
model = model.to(device)
layer_num = 9
path = sys.argv[1]
audio, sampling_rate = sf.read(path)
assert sampling_rate == 16000
with torch.no_grad():
    audio = [torch.from_numpy(audio).float().to(device)]
    states = model(audio)["hidden_states"]
    feature = states[layer_num].squeeze(0)
    feature = feature.cpu().numpy()
reg = load(sys.argv[2])
pred = reg.predict(feature)
np.save(sys.argv[3], pred)

