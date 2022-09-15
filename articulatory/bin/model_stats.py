#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained generator."""

import argparse
import logging
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from articulatory.datasets import MelDataset
from articulatory.datasets import MelSCPDataset
from articulatory.datasets import ArtDataset
from articulatory.datasets import ArtSCPDataset
from articulatory.utils import load_model
from articulatory.utils import read_hdf5
from articulatory.models import NSFA2WModel, A2MGenerator2, HiFiGANGenerator


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    # model = NSFA2WModel(**generator_params).cuda(1)
    # model = A2MGenerator2(**generator_params).cuda(1)
    model = HiFiGANGenerator(**generator_params).cuda(1)
    # art_lens = [i*16 for i in range(26, 1, -5)]
    art_lens = [i*16 for i in range(1, 26, 5)]
    for al in art_lens:
        # x = torch.ones(1, 30, al+1).cuda(1)
        # x = torch.ones(1, 30, al).cuda(1)
        x = torch.ones(1, 80, al).cuda(1)
        t0 = time.time()
        with torch.no_grad():
            o = model(x)
        t1 = time.time()
        print(x.shape[2], t1-t0)
    '''
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )
    '''
