# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Data samplers."""

import json
import logging
import numpy as np
import os
import random
import string
import torch


class SizeAwareSampler(torch.utils.data.Sampler):
    """Returns a batch with the specified total length.

    from David Gaddy's
    https://github.com/dgaddy/silent_speech/blob/main/read_emg.py
    """
    def __init__(self, audio_lens, max_len=2000):
        self.audio_lens = audio_lens
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.audio_lens)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            length = self.audio_lens[idx]
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch
