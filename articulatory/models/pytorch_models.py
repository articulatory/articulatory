#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""PyTorch models."""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from articulatory.layers import HiFiGANResidualBlock as ResidualBlock
from articulatory.layers import WNConv1d, PastFCEncoder
from articulatory.utils import read_hdf5


class BiGRU(nn.Module): #Do we need to generate pitch(?)
    def __init__(self, in_channels=80, hidden_size=256, dropout=0.3, out_channels=1, \
            use_ar=False, ar_input=512, ar_hidden=256, ar_output=128, ar_channels=None, use_tanh=False, \
            use_spk_emb=False, spk_emb_size=32, spk_emb_hidden=32):
        super().__init__()
        self.gru1 = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            # input and output tensors are provided as (batch, seq, feature)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size*2, 128), nn.Dropout(p=dropout))
        self.bn = nn.BatchNorm1d(128)
        if not use_tanh:
            self.fc2 = nn.Linear(128, out_channels)
        else:
            self.fc2 = nn.Sequential(nn.Linear(128, out_channels), nn.Tanh())
        self.use_ar = use_ar
        if use_ar:
            self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)
        self.use_spk_emb = use_spk_emb
        if use_spk_emb:
            self.spk_fc = torch.nn.Linear(spk_emb_size, spk_emb_hidden)

    def forward(self, mels, mask=None, spk_id=None, spk=None, ar=None, ph=None):
        """
        Args:
            mels: N, C_mel, T_mel
            spk: N, C_spk
        
        Return:
            ema: N, C_ema, T_ema
        """
        if self.use_ar:
            ar_feats = self.ar_model(ar)  # (batchsize, ar_output)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, mels.shape[2])  # (batchsize, ar_output, length)
            mels = torch.cat((mels, ar_feats), dim=1)
        if self.use_spk_emb:
            cspk = self.spk_fc(spk)
            cspk = cspk.unsqueeze(2).repeat(1, 1, mels.shape[2])
            mels = torch.cat((mels, cspk), dim=1)
        mels = mels.transpose(1, 2)  # (N, T, C)
        output, hn = self.gru1(mels)
            # output: (N, T, D*H_out), D=2 for bidir
        output = self.dropout1(output)
        output, hn = self.gru2(output)
        output = self.dropout2(output)
        output = self.fc1(output)
        output = output.transpose(1, 2)  # (N, C, T)
        output = self.bn(output).transpose(1, 2)  # (N, T, C)
        output = self.fc2(output).transpose(1, 2)  # (N, C, T)
        return output

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def inference(self, c, normalize_before=True, ar=None, spk=None):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(downsample_scales), out_channels).

        """
        if len(c.shape) == 3:  # TODO make better
            c = c.transpose(1, 2)
            c = c[0]
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.unsqueeze(0).transpose(1, 2), ar=ar, spk=spk)
        return c.transpose(1, 2).squeeze(0)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")
